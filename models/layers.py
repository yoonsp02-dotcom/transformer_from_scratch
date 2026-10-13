import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import DROPOUT, causal_mask
except ImportError:
    from utils import DROPOUT, causal_mask

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, model_dim):
        super().__init__()
        self.max_len = max_len
        self.model_dim = model_dim
        pe = torch.zeros(max_len, model_dim)

        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.pow(10000, -1 * torch.arange(0, model_dim, 2) / model_dim)

        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.Wq = nn.Linear(model_dim, model_dim)
        self.Wk = nn.Linear(model_dim, model_dim)
        self.Wv = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (b, Tx, model_dim) -> (b, h, Tx, head_dim)
        b, Tx, _ = x.shape
        x = x.view(b, Tx, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x):
        # (b, h, Tx, head_dim) -> (b, Tx, model_dim)
        b, h, Tx, head_dim = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(b, Tx, h * head_dim)
        return x
    
    def forward(self, q_base, k_base=None, v_base=None, self_masking=False, key_padding_mask=None):
        if k_base is None:
            k_base = q_base
        if v_base is None:
            v_base = q_base
        
        Q = self.Wq(q_base)
        K = self.Wk(k_base)
        V = self.Wv(v_base)

        # (b, h, T, head_dim)
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # (b, h, TQ, TK)
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))
        scores /= math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:,None,None,:], float("-inf"))

        # self_masking
        if self_masking is True:
            _, _, TQ, TK = scores.shape # TQ == TK
            mask = causal_mask(TQ).to(device=scores.device)
            scores = scores.masked_fill(mask, float('-inf'))

        # (b, h, TQ, TK)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (b, h, TQ, head_dim)
        attn = torch.matmul(attn, V_heads)

        # (b, TQ, model_dim)
        out = self.out(self._merge_heads(attn))
        return out

class RMSNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-8):
        super().__init__()
        self.model_dim = model_dim
        self.eps = eps
        self.g = nn.Parameter(torch.ones(model_dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        x = x / (rms + self.eps)
        return x * self.g

class SwiGLUFFN(nn.Module):
    """
    x -> gate = silu(W_gate x)
         up   = W_up x
         h    = gate * up
         h    = dropout(h)
         out  = W_down h
         out  = dropout(out)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout_p: float = DROPOUT,
        bias: bool = False,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = gate * up
        h = self.dropout1(h)
        out = self.down_proj(h)
        out = self.dropout2(out)
        return out
