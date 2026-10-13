import torch
from transformers import AutoTokenizer

# hyperparameters
MODEL_DIM = 512
NUM_HEADS = 8
BATCH_SIZE = 32
DROPOUT = 0.1
FF_DIM = 1536
ENCODER_LAYER_NUM = 6
DECODER_LAYER_NUM = 6
SRC_LANG = 'de'
TGT_LANG = 'en'
MAX_LEN = 128
BEAM_SIZE = 4
SPLIT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", use_fast=True)
# special tokens
special = {
    "pad": tokenizer.pad_token_id,
    "eos": tokenizer.eos_token_id,
    "unk": tokenizer.unk_token_id
}
PAD_ID = special["pad"]
EOS_ID = special["eos"]
UNK_ID = special["unk"]

def causal_mask(dim):
    mask = torch.triu(torch.ones(dim, dim, dtype=torch.bool), diagonal=1)
    return mask
