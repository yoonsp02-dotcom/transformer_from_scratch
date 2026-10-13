import torch.nn as nn

try:
    from .layers import MultiHeadAttention, RMSNorm, SwiGLUFFN
except ImportError:
    from layers import MultiHeadAttention, RMSNorm, SwiGLUFFN

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = RMSNorm(model_dim=model_dim)
        self.norm2 = RMSNorm(model_dim=model_dim)
        self.ff = SwiGLUFFN(input_dim=model_dim, hidden_dim=d_ff, output_dim=model_dim, dropout_p=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = x + self.dropout(self.mha(self.norm1(x), key_padding_mask=src_key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.mmha = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.cmha = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = RMSNorm(model_dim=model_dim)
        self.norm2 = RMSNorm(model_dim=model_dim)
        self.norm3 = RMSNorm(model_dim=model_dim)
        self.ff = SwiGLUFFN(input_dim=model_dim, hidden_dim=d_ff, output_dim=model_dim, dropout_p=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_key, encoder_val, src_key_padding_mask=None, tgt_key_padding_mask=None):
        x = x + self.dropout(self.mmha(self.norm1(x), self_masking=True, key_padding_mask=tgt_key_padding_mask))
        x = x + self.dropout(self.cmha(q_base=self.norm2(x), k_base=encoder_key, v_base=encoder_val, key_padding_mask=src_key_padding_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, d_ff, dropout, num_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(model_dim=model_dim, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layer)
        )

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, model_dim, num_heads, d_ff, dropout, num_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(model_dim=model_dim, num_heads=num_heads, d_ff=d_ff, dropout=dropout)for _ in range(num_layer)
        )

    def forward(self, x, encoder_key, encoder_val, src_key_padding_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            x = layer(
                x,
                encoder_key,
                encoder_val,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        return x
