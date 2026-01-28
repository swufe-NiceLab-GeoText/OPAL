from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads")

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz = query.size(0)

        q = self.q_proj(query).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, self.dim)
        out = self.out_proj(out)
        out = self.layer_norm(out + query)

        if return_attention:
            attn_weights = attn.squeeze(-1).squeeze(-1)
            return out, attn_weights

        return out, None


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


