import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention2D(nn.Module):
    """
    A lightweight cross-modal attention block that injects text semantics into
    2D visual features.

    - Visual features: (B, C, H, W)
    - Text features: (B, C) or (B, N, C) where N is number of text/object tokens.

    Implementation details:
    - Flattens spatial dims into tokens and uses MultiheadAttention with visual
      queries (HW tokens) attending to a compact set of text keys/values.
    - If text is (B, N, C), it is reduced to a small set by mean pooling (N->1)
      to limit compute and keep this layer cheap.
    - Residual + LayerNorm on the token dimension.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.ln = nn.LayerNorm(dim)
        # Optional output projection and gating for stability
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: (B, C, H, W)
            text: (B, C) or (B, N, C)
        Returns:
            (B, C, H, W) updated visual features
        """
        B, C, H, W = visual.shape

        # Prepare queries from visual tokens
        x = visual.flatten(2).transpose(1, 2)  # (B, HW, C)
        # LayerNorm expects weight/input dtypes to match; always use a dtype-safe path
        x_ln = F.layer_norm(
            x.float(),
            normalized_shape=(C,),
            weight=self.ln.weight.float() if self.ln.weight is not None else None,
            bias=self.ln.bias.float() if self.ln.bias is not None else None,
            eps=self.ln.eps,
        ).to(dtype=x.dtype)
        q = x_ln.transpose(0, 1)  # (HW, B, C)

        # Prepare keys/values from text
        if text.dim() == 2:
            t = text[:, None, :]  # (B, 1, C)
        elif text.dim() == 3:
            # Reduce along N to a single context token (mean pool)
            t = text.mean(dim=1, keepdim=True)  # (B, 1, C)
        else:
            raise ValueError(f"Unsupported text shape: {text.shape}")

        k = t.transpose(0, 1)  # (1, B, C)
        v = k  # share params

        # Cross attention: visual queries attend to text K/V
        # Align q/k/v dtypes with attention parameters when autocast is not active
        attn_weight_dtype = next(self.attn.parameters()).dtype
        q_cast = q.to(dtype=attn_weight_dtype)
        k_cast = k.to(dtype=attn_weight_dtype)
        v_cast = v.to(dtype=attn_weight_dtype)
        attn_out, _ = self.attn(q_cast, k_cast, v_cast)  # (HW, B, C)
        # Cast back to original query dtype prior to projection
        attn_out = attn_out.to(dtype=q.dtype)
        attn_out = attn_out.transpose(0, 1)  # (B, HW, C)
        # Linear expects input dtype to match weight dtype (commonly float32)
        proj_w_dtype = self.proj.weight.dtype
        attn_out_proj = self.proj(attn_out.to(dtype=proj_w_dtype))
        attn_out = attn_out_proj.to(dtype=attn_out.dtype)

        # Residual with gating
        # Ensure residual addition uses consistent dtype
        attn_out = attn_out.to(dtype=x.dtype)
        gamma = self.gamma.to(dtype=x.dtype)
        x = x + gamma * attn_out

        # Restore spatial shape
        out = x.transpose(1, 2).reshape(B, C, H, W)
        return out
