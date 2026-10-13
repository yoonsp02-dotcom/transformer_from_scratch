from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm

from models.preprocess import move_batch_to_device
from models.transformer import Transformer
from models.utils import DEVICE, EOS_ID, MAX_LEN, PAD_ID, tokenizer


CHECKPOINT_PATH = Path("checkpoints/transformer_best.pt")
MAX_NEW_TOKENS = MAX_LEN


@torch.no_grad()
def greedy_decode(model, src_ids, src_key_padding_mask, max_new_tokens):
    B = src_ids.size(0)
    generated = torch.full((B, 1), PAD_ID, dtype=torch.long, device=src_ids.device)
    finished = torch.zeros(B, dtype=torch.bool, device=src_ids.device)

    for _ in range(max_new_tokens):
        logits = model(
            src_ids=src_ids,
            tgt_ids=generated,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=None,
        )
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        finished = finished | (next_token == EOS_ID)
        if finished.all():
            break

    outputs = []
    for seq in generated[:, 1:].tolist():
        if EOS_ID is not None and EOS_ID in seq:
            seq = seq[: seq.index(EOS_ID)]
        while seq and seq[-1] == PAD_ID:
            seq.pop()
        outputs.append(seq)
    return outputs


@torch.no_grad()
def get_bleu_score(dataloader):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        pad_id=PAD_ID,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    pred_texts = []
    ref_texts = []
    pbar = tqdm(dataloader, desc="bleu-greedy")

    for batch in pbar:
        batch = move_batch_to_device(batch, DEVICE)
        src_ids = batch["input_ids"]
        src_key_padding_mask = batch["attention_mask"] == 0

        pred_ids = greedy_decode(
            model=model,
            src_ids=src_ids,
            src_key_padding_mask=src_key_padding_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        labels = batch["labels"].clone()
        labels[labels == -100] = PAD_ID

        ref_ids = []
        for seq in labels.tolist():
            if EOS_ID is not None and EOS_ID in seq:
                seq = seq[: seq.index(EOS_ID)]
            while seq and seq[-1] == PAD_ID:
                seq.pop()
            ref_ids.append(seq)

        pred_texts.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
        ref_texts.extend(tokenizer.batch_decode(ref_ids, skip_special_tokens=True))

    bleu = sacrebleu.corpus_bleu(pred_texts, [ref_texts])
    return bleu.score
