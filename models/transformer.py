import torch.nn as nn

try:
    from .encoder_decoder import Decoder, Encoder
    from .layers import PositionalEncoding, SwiGLUFFN
    from .utils import (
        DECODER_LAYER_NUM,
        DROPOUT,
        ENCODER_LAYER_NUM,
        FF_DIM,
        MAX_LEN,
        MODEL_DIM,
        NUM_HEADS,
        PAD_ID,
    )
except ImportError:
    from encoder_decoder import Decoder, Encoder
    from layers import PositionalEncoding, SwiGLUFFN
    from utils import (
        DECODER_LAYER_NUM,
        DROPOUT,
        ENCODER_LAYER_NUM,
        FF_DIM,
        MAX_LEN,
        MODEL_DIM,
        NUM_HEADS,
        PAD_ID,
    )

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size,
                 pad_id = PAD_ID, 
                 model_dim = MODEL_DIM,
                 n_heads = NUM_HEADS,
                 num_encoder_layers = ENCODER_LAYER_NUM,
                 num_decoder_layers = DECODER_LAYER_NUM,
                 max_len = MAX_LEN,
                 d_ff = FF_DIM,
                 dropout=DROPOUT):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, model_dim, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, model_dim, padding_idx=pad_id)

        self.pos_embed = PositionalEncoding(max_len, model_dim)
        self.drop = nn.Dropout(dropout)

        self.encoder = Encoder(
            model_dim=model_dim,
            num_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layer=num_encoder_layers,
        )
        self.decoder = Decoder(
            model_dim=model_dim,
            num_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layer=num_decoder_layers,
        )

        self.lm_head = SwiGLUFFN(
            input_dim=model_dim,
            hidden_dim=d_ff,
            output_dim=tgt_vocab_size,
            dropout_p=dropout,
        )

        self.model_dim = model_dim
        self.pad_id = pad_id

    def forward(self, src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask):
        src = self.src_embed(src_ids)
        tgt = self.tgt_embed(tgt_ids)

        src = self.drop(self.pos_embed(src))
        tgt = self.drop(self.pos_embed(tgt))

        encoder_output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            encoder_output,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.lm_head(decoder_output)
        return logits
