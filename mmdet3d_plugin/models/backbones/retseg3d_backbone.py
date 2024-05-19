from typing import List, Optional
import functools
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.pytorch import SparseConvTensor
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS
from ..utils.retention import Retention


class ResSubMBlock(SparseModule):
    """ Residual Sparse Block with SubMConv3d"""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 norm_fn,
                 indice_key: Optional[str] = None
                 ):
        super().__init__()

        # conv branch with 2 submconv3d
        self.conv_branch = spconv.SparseSequential(
            norm_fn(input_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key),
            norm_fn(output_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key),
        )

        # residual branch
        if input_channels == output_channels:
            self.res_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.res_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    bias=False
                )
            )

    def forward(self, inp: SparseConvTensor) -> SparseConvTensor:
        """
        Args:
            inp (spconv.SparseTensor)
        Returns:
            output (spconv.SparseTensor)
        """
        residual = spconv.SparseConvTensor(
            inp.features,
            inp.indices,
            inp.spatial_shape,
            inp.batch_size
        )
        output = self.conv_branch(inp)
        output = output.replace_feature(
            output.features + self.res_branch(residual).features)
        return output


class EncoderDownBlock(nn.Module):
    """ The Encoder Down Block of RetSeg3D
    It consists of ResSubMBlock + Retention + SparseConv Down sampling
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 block_reps: int,
                 norm_fn,
                 ret_cfg: dict = dict(num_heads=2,
                                      window_shape=[6, 6, 6],
                                      sparse_shape=[2048, 2048, 128],
                                      group_size=69,
                                      direction=["x"],
                                      drop_path=0.0),
                 indice_key_id: int = None,
                 grad_ckpt_layers: List[int] = [],
                 downstride: bool = True):

        super().__init__()

        self.indice_key_id = indice_key_id
        self.grad_ckpt_layers = grad_ckpt_layers
        self.downstride = downstride

        # ResSubMBlock with 2 SubMConv3D
        ressubmblocks_enc = {
            f'ressubmblock_enc{i}': ResSubMBlock(input_channels,
                                                 input_channels,
                                                 norm_fn,
                                                 indice_key=f'subm'
                                                            f'{indice_key_id}')
            for i in range(block_reps)
        }
        ressubmblocks_enc = OrderedDict(ressubmblocks_enc)
        self.ressubmblocks_enc = spconv.SparseSequential(ressubmblocks_enc)

        # Put Transformer here
        if self.indice_key_id in [1,2,3,4,5]:
            self.retention_block = Retention(
                embed_dim=input_channels,
                **ret_cfg
            )

        # SparseConv Down sampling
        if self.downstride:
            self.sparseconv_down = spconv.SparseSequential(
                norm_fn(input_channels),
                nn.ReLU(),
                spconv.SparseConv3d(
                    input_channels,
                    output_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key=f'spconv{indice_key_id}'
                )
            )

    def forward(self, inp: SparseConvTensor) -> SparseConvTensor:
        """
        Args:
            inp (SparseConvTensor)
        Returns:
            (Tuple(SparseConvTensor)): first is final output after down
            sampling and second is before down sampling
        """
        output = self.ressubmblocks_enc(inp)

        # check grad_ckpt
        if self.indice_key_id in [1,2,3,4,5]:
            if self.indice_key_id in self.grad_ckpt_layers:
                def ret_block_fn(inp_, coords_, bs_):
                    return self.retention_block(inp_, coords_, bs_)

                from torch.utils.checkpoint import checkpoint
                retblock_feats = checkpoint(ret_block_fn,
                                            output.features,
                                            output.indices,
                                            output.batch_size)
            else:
                retblock_feats = self.retention_block(output.features,
                                                      output.indices,
                                                      output.batch_size)
            output = output.replace_feature(retblock_feats)

        # down sampling
        if self.downstride:
            output_down = self.sparseconv_down(output)
            return output_down, output
        else:
            return output


class DecoderUpBlock(nn.Module):
    """
    The Decoder block of RetSeg3D
    it consists of SparseInvConvModule + concat lateral feat from encoder +
    ResSubMBlock
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 block_reps: int,
                 norm_fn,
                 indice_key_id: int = None,
                 ):
        super().__init__()

        self.indice_key_id = indice_key_id

        self.sparseconv_up = spconv.SparseSequential(
            norm_fn(input_channels),
            nn.ReLU(),
            spconv.SparseInverseConv3d(
                input_channels,
                output_channels,
                kernel_size=2,
                bias=False,
                indice_key=f'spconv{indice_key_id}'
            )
        )

        ressubmblocks_dec = {
            f'ressubmblocks_dec{i}': ResSubMBlock(
                output_channels * (2-i),
                output_channels,
                norm_fn,
                indice_key=f'subm{indice_key_id}'
            ) for i in range(block_reps)
        }
        ressubmblocks_dec = OrderedDict(ressubmblocks_dec)
        self.ressubmblocks_dec = spconv.SparseSequential(ressubmblocks_dec)

    def forward(self,
                inp: SparseConvTensor,
                inp_up: SparseConvTensor) -> SparseConvTensor:
        """
        Args:
            inp (SparseConvTensor): the input to upsample
            inp_up (SparseConvTensor): the lateral feature from upper layer of
                encoder
        """
        # InvConv
        output = self.sparseconv_up(inp)

        # concat lateral feat from encoder
        output = output.replace_feature(torch.cat([
            inp_up.features, output.features
        ], dim=1))

        # DecoderUpBlocks
        output = self.ressubmblocks_dec(output)

        return output


@MODELS.register_module()
class RetSeg3DBackbone(BaseModule):
    """Backbone for RetSeg3D

     Args:
        input_channels (int): input channels of spconvtensor
        encoder_channels (List[int]): channels of various layers of encoder
        block_reps (int): number of block repetitions of ResSubMBlock
     """

    def __init__(self,
                 input_channels: int,
                 encoder_channels: List[int],
                 block_reps: int = 2,
                 ret_cfg: dict = dict(
                     num_heads=[2, 2, 4, 4, 4],
                     window_shape=[[6, 6, 6], [12, 12, 12], [24, 24, 24],
                                   [48, 48, 16], [48, 48, 8]],
                     sparse_shape=[[2048, 2048, 128], [1024, 1024, 64],
                                   [512, 512, 32], [256, 256, 16],
                                   [128, 128, 8]],
                     group_size=[184, 1470, 11750, 31330, 18432],
                     direction=[["x"], ["y"], ["x"], ["y"], ["x"]],
                     drop_path_rate=0.3
                 ),
                 grad_ckpt_layers: List[int] = [],
                 grid_size: List[int] = [],
                 init_cfg=None):
        super(RetSeg3DBackbone, self).__init__(init_cfg=init_cfg)

        self.grid_size = grid_size

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        dpr = [x.item() for x in torch.linspace(0,
                                                ret_cfg['drop_path_rate'],
                                                5)]

        # input conv
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels,
                encoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            )
        )

        # Encoder down blocks
        self.encoder_down_block_list = torch.nn.ModuleList()
        self.decoder_up_block_list = torch.nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder_down_block_list.append(
                EncoderDownBlock(
                    input_channels=encoder_channels[i],
                    output_channels=encoder_channels[i + 1],
                    block_reps=block_reps,
                    norm_fn=norm_fn,
                    ret_cfg=dict(
                        num_heads=ret_cfg['num_heads'][i],
                        window_shape=ret_cfg['window_shape'][i],
                        sparse_shape=ret_cfg['sparse_shape'][i],
                        group_size=ret_cfg['group_size'][i],
                        direction=ret_cfg['direction'][i],
                        drop_path=dpr[i],
                    ),
                    indice_key_id=i + 1,
                    grad_ckpt_layers=grad_ckpt_layers,
                    downstride=True,
                )
            )
            self.decoder_up_block_list.append(
                DecoderUpBlock(
                    input_channels=encoder_channels[-1-i],
                    output_channels=encoder_channels[-2-i],
                    block_reps=block_reps,
                    norm_fn=norm_fn,
                    indice_key_id=4-i,
                )
            )
        # additional encoder block at last after down sampling
        self.encoder_last_block = EncoderDownBlock(
            input_channels=encoder_channels[-1],
            output_channels=encoder_channels[-1],
            block_reps=block_reps,
            norm_fn=norm_fn,
            ret_cfg=dict(
                num_heads=ret_cfg['num_heads'][-1],
                window_shape=ret_cfg['window_shape'][-1],
                sparse_shape=ret_cfg['sparse_shape'][-1],
                group_size=ret_cfg['group_size'][-1],
                direction=ret_cfg['direction'][-1],
                drop_path=dpr[-1],
            ),
            indice_key_id=len(encoder_channels),
            grad_ckpt_layers=grad_ckpt_layers,
            downstride=False,
        )

        # additional Norm + ReLU after decoder block as last decoder block
        # finished with SubMConv3D
        self.decoder_last_layer = spconv.SparseSequential(
            norm_fn(encoder_channels[0]),
            nn.ReLU()
        )

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self,
                voxel_features: Tensor,
                coords: Tensor,
                batch_size: int) -> SparseConvTensor:

        # coors as [b, x, y, z]
        # initialize SparseConvTensor for input
        coords = coords.int()
        input_sptensor = SparseConvTensor(voxel_features,
                                          coords,
                                          np.array(self.grid_size),
                                          batch_size)

        # input conv with SubMConv3D
        output = self.input_conv(input_sptensor)

        # Encoder down blocks
        output_lateral_list = []
        output_down = output
        for encoder_down_block in self.encoder_down_block_list:
            output_down, output_lateral = encoder_down_block(output_down)
            output_lateral_list.append(output_lateral)

        # additional encoder block at last after down sampling
        output_down = self.encoder_last_block(output_down)

        # Decoder up blocks
        output_up = output_down
        for i, decoder_up_block in enumerate(self.decoder_up_block_list):
            output_up = decoder_up_block(output_up, output_lateral_list[-1-i])

        # last decoder layer
        output = self.decoder_last_layer(output_up)

        return output
