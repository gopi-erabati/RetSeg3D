from typing import List, Dict, Tuple, Any

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn.bricks import DropPath
from .retention_utils import (MultiScaleRetention,
                              MultiScaleRetentionSoftmax, RMSNorm, GLU)


class RetentionEncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            activation: str,
            group_size: int,
            drop_path: float = 0.0,
            normalize_before=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize_before = normalize_before
        self.ffn_dim = self.embed_dim * 2
        self.num_heads = num_heads
        self.group_size = group_size

        self.retention = MultiScaleRetentionSoftmax(embed_dim, num_heads)
        self.retention_layer_norm = RMSNorm(embed_dim, eps=1e-6)
        self.ffn = GLU(embed_dim, self.ffn_dim, activation)
        self.final_layer_norm = RMSNorm(embed_dim, eps=1e-6)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor, pe: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): input voxels of shape (BM, d)
            pe (torch.Tensor): pos encoding of shape (BM, d)
            coords (torch.Tensor): input voxel coordinates of shape (BM, 4)
        Returns:
            output (torch.Tensor): shape (BM, d)
        """
        # Retention
        residual = x

        if self.normalize_before:
            x = self.retention_layer_norm(x)

        # Grouping
        size = x.shape[0]  # BM
        num_groups = int(math.ceil(size / self.group_size))

        x = x.view(num_groups, self.group_size, -1)
        pe = pe.view(num_groups, self.group_size, -1)
        coords = coords.view(num_groups, self.group_size, -1)

        # decay as in retention depending on manhattan distance between voxels
        decay = torch.log(1 - 2 ** (-5 - torch.arange(self.num_heads,
                                                      dtype=torch.float)))
        # ablation with uniform decay for every head , decay = 127/128
        # decay = torch.log(torch.ones((self.num_heads, ), dtype=torch.float)
        #                   * 127/128)
        decay = decay.to(x.device)
        # (h, )
        dist_2d_xn = coords[:, :, 1].unsqueeze(2)  # (n_g, g, 1)
        dist_2d_xm = coords[:, :, 1].unsqueeze(1)  # (n_g, 1, g)
        dist_2d_yn = coords[:, :, 2].unsqueeze(2)  # (n_g, g, 1)
        dist_2d_ym = coords[:, :, 2].unsqueeze(1)  # (n_g, 1, g)
        dist_2d_zn = coords[:, :, 3].unsqueeze(2)  # (n_g, g, 1)
        dist_2d_zm = coords[:, :, 3].unsqueeze(1)  # (n_g, 1, g)
        dist_2d = (torch.abs(dist_2d_xn - dist_2d_xm) + torch.abs(dist_2d_yn
                                                                 -
                                                                 dist_2d_ym)
                   + torch.abs(dist_2d_zn - dist_2d_zm))
        # (n_g, g, g)
        decay_mask = torch.exp(dist_2d * decay[:, None, None, None])
        # (h, n_g, g, g)
        decay_mask = torch.nan_to_num(decay_mask)
        # ablation w/o decay, i.e., decay = 1
        # decay_mask = torch.ones((self.num_heads, num_groups,
        #                          self.group_size, self.group_size),
        #                         dtype=torch.float).to(x.device)
        decay_mask = decay_mask / decay_mask.sum(dim=-1, keepdim=True).sqrt()
        decay_mask = decay_mask.permute(1, 0, 2, 3)  # (n_g, h, g, g)

        x = self.retention(x, pe, decay_mask)  # (n_g, g, d)
        x = x.view(num_groups * self.group_size, -1)  # (n_g*g, -1) (BM, d)

        x = residual + self.drop_path(x)  # (BM, d)

        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        # FFN
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.ffn(x)

        x = residual + self.drop_path(x)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x  # (BM, d)


class RetentionEncoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            activation: str,
            group_size: int,
            direction: List[str] = ["x"],
            drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.direction = direction
        self.encoder_block = nn.ModuleList()
        for _ in range(len(direction)):
            layer = RetentionEncoderLayer(
                embed_dim,
                num_heads,
                activation,
                group_size=group_size,
                drop_path=drop_path,
            )
            self.encoder_block.append(layer)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, mappings: Dict[str,
    Any], coords: torch.Tensor) -> torch.Tensor:
        for k, name in enumerate(self.direction):
            indices = mappings[name]  # (BM,)
            x[indices] = \
                self.encoder_block[k](x[indices][mappings["flat2win"]],
                                      pe[indices][mappings["flat2win"]],
                                      coords[indices][mappings['flat2win']])[
                    mappings["win2flat"]
                ]

        return x  # (BM, d)


@torch.inference_mode()
def get_window_coors_shift(coords: Tensor, sparse_shape: List[int],
                           window_shape: List[int],
                           shifted: bool) -> Tuple[int]:
    n, m, l = sparse_shape
    n2, m2, l2 = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.
    l1 = int(np.ceil(l / l2) + 1)

    if shifted:
        shift_x, shift_y, shift_z = (n2 // 2, m2 // 2, l2 // 2)
        x = coords[:, 1] + shift_x
        y = coords[:, 2] + shift_y
        z = coords[:, 3] + shift_z
    else:
        x = coords[:, 1]
        y = coords[:, 2]
        z = coords[:, 3]

    x1 = x // n2
    y1 = y // m2
    z1 = z // l2
    x2 = x % n2
    y2 = y % m2
    z2 = z % l2

    return (3 * n2, 3 * m2, 3 * l2, 3 * n1, 3 * m1, 3 * l1, x1, y1, z1, x2,
            y2, z2)


class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            sparse_shape,
            group_size,
    ) -> None:
        super().__init__()
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.group_size = group_size

    def forward(self, coords: torch.Tensor, batch_size: int) -> Dict[
        str, torch.Tensor]:
        coords = coords.long()

        _, num_per_batch = torch.unique(coords[:, 0], sorted=False,
                                        return_counts=True)
        # (B, )
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        # (B+1, )
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[
                                              :-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0),
                                      (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1]).to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1]).to(coords.device)
        for i in range(batch_size):
            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )
            if num_per_batch[i] != num_per_batch_p[i]:
                flat2win[
                batch_start_indices_p[i + 1]
                - self.group_size
                + (num_per_batch[i] % self.group_size): batch_start_indices_p[
                    i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size):
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    ]
            flat2win[
            batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}
        for shifted in [False, True]:
            (
                n2,
                m2,
                l2,
                n1,
                m1,
                l1,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2
            ) = get_window_coors_shift(coords, self.sparse_shape,
                                       self.window_shape, shifted=shifted)

            vx = ((n1 * y1 + n1 * m1 * z1 + (-1) ** y1 * x1) * n2 * m2 * l2 +
                  (-1) ** y1 * (m2 * x2 + n2 * m2 * z2 + (-1) ** x2 * y2))
            vx += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[
                1] * self.sparse_shape[2] * 10
            vy = ((m1 * z1 + m1 * l1 * x1 + (-1) ** z1 * y1) * m2 * l2 * n2 +
                  (-1) ** z1 * (l2 * y2 + m2 * l2 * x2 + (-1) ** y2 * z2))
            vy += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[
                1] * self.sparse_shape[2] * 10
            vz = ((l1 * x1 + l1 * n1 * y1 + (-1) ** x1 * z1) * l2 * n2 * m2 +
                  (-1) ** x1 * (n2 * z2 + l2 * n2 * y2 + (-1) ** z2 * x2))
            vz += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[
                1] * self.sparse_shape[2] * 10
            _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
            _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)
            _, mappings["z" + ("_shift" if shifted else "")] = torch.sort(vz)

        return mappings


class PositionalEmbedding(nn.Module):
    """
    3D Positional Embedding for voxel coordinates
    """

    def __init__(self,
                 feat_dim: int,
                 pos_temperature: int,
                 ):
        super().__init__()
        self.pos_length = None
        self.feat_dim = feat_dim
        self.pos_temperature = pos_temperature

    def forward(self, coords: Tensor, dtype):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]  # (BM, )

        inv_freq = self.inv_freq  # (d/2, )

        # [num_tokens, pos_length]
        pex = x[:, None] / inv_freq()[None, :]  # (BM, d/3)
        pey = y[:, None] / inv_freq()[None, :]  # (BM, d/3)
        pez = z[:, None] / inv_freq()[None, :]  # (BM, d/3)

        # [num_tokens, pos_length]
        pex = torch.stack([pex[:, ::2].sin(), pex[:, 1::2].cos()],
                          dim=-1).flatten(1)  # (BM, d/3)
        pey = torch.stack([pey[:, ::2].sin(), pey[:, 1::2].cos()],
                          dim=-1).flatten(1)  # (BM, d/3)
        pez = torch.stack([pez[:, ::2].sin(), pez[:, 1::2].cos()],
                          dim=-1).flatten(1)  # (BM, d/3)
        pe = torch.cat([pex, pey, pez], dim=-1).to(dtype)  # (BM, d)

        gap = self.feat_dim - pe.size(1)
        if gap < 0:
            pe = pe[:, :self.feat_dim]

        return pe  # (BM, d)

    def inv_freq(self):
        ndim = 3
        pos_length = int(np.ceil(self.feat_dim / (ndim * 2)) * 2)  # feat_dim/3

        # [pos_length]
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device="cuda")
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        return inv_freq


class Retention(nn.Module):
    """
    Retention Block used in RetSeg3D
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_shape: List[int],
                 sparse_shape: List[int],
                 pos_temperature: int = 10000,
                 group_size: int = 69,
                 activation="gelu",
                 direction: List[str] = ["x"],
                 drop_path: float = 0.0,
                 ):
        super().__init__()

        self.embedding = PositionalEmbedding(
            embed_dim,
            pos_temperature,
        )

        self.mapping = FlattenedWindowMapping(
            window_shape=window_shape,
            sparse_shape=sparse_shape,
            group_size=group_size,
        )

        self.retention_block = RetentionEncoderBlock(
            embed_dim,
            num_heads,
            activation,
            group_size,
            direction,
            drop_path
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inp: Tensor, coords: Tensor, batch_size: int):

        pe = self.embedding(coords, inp.dtype)
        mappings = self.mapping(coords, batch_size)
        output = self.retention_block(inp, pe, mappings, coords)

        return output
