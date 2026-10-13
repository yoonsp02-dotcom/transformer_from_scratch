import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
try:
    from .utils import BATCH_SIZE, MAX_LEN, SRC_LANG, TGT_LANG, SPLIT_SEED, special, tokenizer
except ImportError:
    from utils import BATCH_SIZE, MAX_LEN, SRC_LANG, TGT_LANG, SPLIT_SEED, special, tokenizer

def preprocess_function(examples):
    src_texts = [x[SRC_LANG] for x in examples["translation"]]
    tgt_texts = [x[TGT_LANG] for x in examples["translation"]]

    model_inputs = tokenizer(src_texts, max_length=MAX_LEN, truncation=True, padding=False)
    labels = tokenizer(tgt_texts, max_length=MAX_LEN, truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    '''
    "input_ids":
    "attention_mask": 
    "labels":
    '''
    return model_inputs

def collate_fn(batch):
    '''
    batch
    [
      {"input_ids": [...], "attention_mask": [...], "labels": [...]},
      {"input_ids": [...], "attention_mask": [...], "labels": [...]},
    ]
    '''
    pad_id = special["pad"]
    decoder_start_id = special["pad"]

    src_list = [x["input_ids"] for x in batch]
    tgt_list = [x["labels"]    for x in batch]
    B = len(src_list)

    # get max_len
    src_max_len = max(len(x) for x in src_list)
    tgt_max_len = max(len(x) for x in tgt_list)

    # allocate padded tensors
    src_padded = torch.full((B, src_max_len), pad_id, dtype=torch.long)
    tgt_padded = torch.full((B, tgt_max_len), pad_id, dtype=torch.long)

    # fill padded tensors
    for i, src in enumerate(src_list):
        src_padded[i, :len(src)] = torch.tensor(src, dtype=torch.long)
    for i, tgt in enumerate(tgt_list):
        tgt_padded[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    # encoder attention mask
    attention_mask = (src_padded != pad_id).long()

    # labels for loss: pad -> -100 (ignore idx)
    labels = tgt_padded.clone()
    labels[labels == pad_id] = -100

    # right shift for teacher forcing
    decoder_input_ids = torch.full((B, tgt_max_len), pad_id, dtype=torch.long)
    decoder_input_ids[:, 0] = decoder_start_id
    if tgt_max_len > 1:
        decoder_input_ids[:, 1:] = tgt_padded[:, :-1]

    # decoder_attention_mask
    decoder_attention_mask = torch.zeros((B, tgt_max_len), dtype=torch.long)
    if tgt_max_len > 0:
        decoder_attention_mask[:, 0] = 1
    if tgt_max_len > 1:
        decoder_attention_mask[:, 1:] = (tgt_padded[:, :-1] != pad_id).long()
    
    return {
        "input_ids": src_padded,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }

def prepare_dataloaders(
    split_seed=SPLIT_SEED,
    train_limit=None,
    dev_limit=None,
    test_limit=None,
):
    train_ds = load_dataset("wmt/wmt19", "de-en", split="train[:100000]")
    valid_ds = load_dataset("wmt/wmt19", "de-en", split="validation")

    valid_ds = valid_ds.shuffle(seed=split_seed)
    mid = len(valid_ds) // 2
    dev_ds = valid_ds.select(range(mid))
    test_ds = valid_ds.select(range(mid, len(valid_ds)))

    if train_limit is not None:
        train_ds = train_ds.select(range(min(train_limit, len(train_ds))))
    if dev_limit is not None:
        dev_ds = dev_ds.select(range(min(dev_limit, len(dev_ds))))
    if test_limit is not None:
        test_ds = test_ds.select(range(min(test_limit, len(test_ds))))

    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    tokenized_dev = dev_ds.map(preprocess_function, batched=True, remove_columns=dev_ds.column_names)
    tokenized_test = test_ds.map(preprocess_function, batched=True, remove_columns=test_ds.column_names)

    train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(tokenized_dev, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders()
