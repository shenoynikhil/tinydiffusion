"""Simple implementation fo a bert style encoder only network

Implemented with
- Diffusion Transformer (DiT) from https://arxiv.org/abs/2212.09748
"""

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(self, hidden_dim: int, n_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, n]
        q, k, v = self.qkv.forward(x).chunk(3, dim=-1)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.n_heads)

        # set shape for scaled_dot_product_attention
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        # apply scaled dot product attention, rearrange and project
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.proj(x)

        return x # [batch_size, seq_len, hidden_dim]


class AdaLNZeroBlock(nn.Module):
    """AdaLNZeroBlock: Ada(ptive) L(ayer) N(orm) Zero(initialized) Block
    From the DiT paper. Basically a linear layer to learn the shift and scale weights.
    Both weight and bias are initialized to zero.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)        


class DiTBlock(nn.Module):
    """DiTBlock"""
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm_msa = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_dim, n_heads, qkv_bias)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        )
        self.adaLN = AdaLNZeroBlock(input_dim=hidden_dim, output_dim=6 * hidden_dim)
    
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN.forward(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm_msa(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """DiT: Diffusion Transformer
    
    Simplifications: 
    - Using a simple 1D positional encoding instead of 2D, RoPE, etc.
    """
    def __init__(
        self,
        patch_size: int,
        num_channels: int,
        image_dim: int,
        n_layers: int = 2,
        n_heads: int = 2,
        hidden_dim: int = 64,
        qkv_bias: bool = True,
    ):
        super().__init__()

        # embedding, positional encoding, block layers, and output layer
        assert image_dim % patch_size == 0

        patch_dim = (patch_size * patch_size * num_channels)
        num_patches = (image_dim // patch_size) ** 2
        h = w = image_dim // patch_size
        
        # set initial patchification
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches), requires_grad=True) # learnable 1D positional encoding
        self.patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size, h=h, w=w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.noise_level_embedding = nn.Linear(1, hidden_dim)
        
        # attention layers
        self.layers = nn.ModuleList([DiTBlock(hidden_dim, n_heads, qkv_bias=qkv_bias) for _ in range(n_layers)])
        
        # after attention
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = AdaLNZeroBlock(input_dim=hidden_dim, output_dim=2 * hidden_dim)
        self.unpatchify = nn.Sequential(
            nn.Linear(hidden_dim, patch_dim),
            Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size, h=h, w=w),
        )
    
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.patch_embedding(x) + self.pos_encoding
        c = self.noise_level_embedding(c) # [batch_size,] -> [batch_size, hidden_dim]
        for layer in self.layers:
            x = layer(x, c)
        
        x = self.norm_final(x)
        shift_final, scale_final = self.adaLN.forward(c).chunk(2, dim=1)
        x = modulate(x, shift_final, scale_final)
        x = self.unpatchify(x)

        return x


if __name__ == "__main__":
    # simple test
    model = DiT(patch_size=7, num_channels=1, image_dim=28, hidden_dim=16)
    x = torch.randn(2, 1, 28, 28) # 2 images of size 1 x 28 x 28
    c = torch.randn(2, 1)
