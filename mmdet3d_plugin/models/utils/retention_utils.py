# Modified from Microsoft torchscale
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


import torch
import torch.nn.functional as F
from torch import nn


def get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    if activation == "swish":
        return F.silu


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.fc2(x)
        x = x.view(x_shape)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class MultiScaleRetention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            gate_fn="swish",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, self.value_dim, bias=False)

        self.out_proj = nn.Linear(self.value_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=1e-6,
                                  elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, pe, decay_mask):
        """
        Args:
            x (torch.Tensor): shape (n_groups, group_size, d) (n_g, g, d)
            pe (torch.Tensor): shape (n_groups, group_size, d) (n_g, g, d)
            decay_mask (torch.Tensor): shape (n_g, h, g, g)
        Returns:
            output (torch.Tensor): shape # (n_g, g, d)
        """
        bsz, num_tokens, embed_dim = x.shape

        q = k = x + pe
        v = x

        q = self.q_proj(q)  # (n_g, g, d)
        k = self.k_proj(k)
        v = self.v_proj(v)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, num_tokens, self.num_heads, self.key_dim).transpose(1,
                                                                            2)
        k = k.view(bsz, num_tokens, self.num_heads, self.key_dim).transpose(1,
                                                                            2)
        v = v.view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1,
                                                                             2)
        # (n_g, h, g, dh)

        qk_mat = q @ k.transpose(-1, -2)
        # (n_g, h, g, dh) @ (n_g, h, dh, g) -> (n_g, h, g, g)
        qk_mat = qk_mat * decay_mask
        # (n_g, h, g, g) * (n_g, h, g, g) -> (n_g, h, g, g)
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1,
                                              keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, v)
        # (n_g, h, g, g) @ (n_g, h, g, dh) -> (n_g, h, g, dh)
        output = output.transpose(1, 2)  # (n_g, g, h, dh)

        output = self.group_norm(output).reshape(bsz, num_tokens,
                                                 self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output  # (n_g, g, d)


class MultiScaleRetentionSoftmax(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            gate_fn="swish",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.value_dim, bias=False)

        self.out_proj = nn.Linear(self.value_dim, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, pe, decay_mask):
        """
        Args:
            x (torch.Tensor): shape (n_groups, group_size, d) (n_g, g, d)
            pe (torch.Tensor): shape (n_groups, group_size, d) (n_g, g, d)
            decay_mask (torch.Tensor): shape (n_g, h, g, g)
        Returns:
            output (torch.Tensor): shape # (n_g, g, d)
        """
        bsz, num_tokens, embed_dim = x.shape

        q = k = x + pe
        v = x

        q = self.q_proj(q)  # (n_g, g, d)
        k = self.k_proj(k)
        v = self.v_proj(v)

        k *= self.scaling
        q = q.view(bsz, num_tokens, self.num_heads, self.key_dim).transpose(1,
                                                                            2)
        k = k.view(bsz, num_tokens, self.num_heads, self.key_dim).transpose(1,
                                                                            2)
        v = v.view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1,
                                                                             2)
        # (n_g, h, g, dh)

        qk_mat = q @ k.transpose(-1, -2)
        # (n_g, h, g, dh) @ (n_g, h, dh, g) -> (n_g, h, g, g)
        attn_weights = F.softmax(qk_mat, dim=-1,
                                 dtype=torch.float32).type_as(qk_mat)
        # (n_g, h, g, g)
        attn_weights = attn_weights * decay_mask
        # (n_g, h, g, g) * (n_g, h, g, g) -> (n_g, h, g, g)
        output = torch.matmul(attn_weights, v)
        # (n_g, h, g, g) @ (n_g, h, g, dh) -> (n_g, h, g, dh)
        output = output.transpose(1, 2).reshape(bsz, num_tokens,
                                                self.head_dim * self.num_heads)
        # (n_g, h, g, dh) -> (n_g, g, h, dh) -> (n_g, g, d)

        output = self.out_proj(output)

        return output  # (n_g, g, d)
