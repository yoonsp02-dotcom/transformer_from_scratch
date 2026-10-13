import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import eval_bleu
from models.preprocess import move_batch_to_device, prepare_dataloaders
from models.transformer import Transformer
from models.utils import DEVICE, MODEL_DIM, PAD_ID, SPLIT_SEED, tokenizer

DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.98
DEFAULT_ADAM_EPS = 1e-9
DEFAULT_WARMUP_STEPS = 4000
DEFAULT_METRICS_NAME = "training_metrics.json"
DEFAULT_SMOKE_TRAIN_SIZE = 1000
DEFAULT_SMOKE_DEV_SIZE = 100
DEFAULT_SMOKE_TEST_SIZE = 100
DEFAULT_SMOKE_TRAIN_STEPS = 4
DEFAULT_SMOKE_EVAL_STEPS = 2


def noam_learning_rate(step_num, model_dim, warmup_steps):
    if step_num <= 0:
        raise ValueError("step_num must be >= 1")
    return (model_dim ** -0.5) * min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    stage_name,
    optimizer=None,
    grad_clip=1.0,
    max_steps=None,
    lr_scheduler_state=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, desc=stage_name)

    for step_idx, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        src_key_padding_mask = batch["attention_mask"] == 0
        tgt_key_padding_mask = batch["decoder_attention_mask"] == 0

        if is_train:
            if lr_scheduler_state is not None:
                lr_scheduler_state["step_num"] += 1
                lr = noam_learning_rate(
                    step_num=lr_scheduler_state["step_num"],
                    model_dim=lr_scheduler_state["model_dim"],
                    warmup_steps=lr_scheduler_state["warmup_steps"],
                ) * lr_scheduler_state["lr_factor"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                lr_scheduler_state["last_lr"] = lr
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(
                batch["input_ids"],
                batch["decoder_input_ids"],
                src_key_padding_mask,
                tgt_key_padding_mask,
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                batch["labels"].reshape(-1),
            )

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        if is_train and lr_scheduler_state is not None:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler_state['last_lr']:.2e}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if max_steps is not None and step_idx >= max_steps:
            break

    return total_loss / max(total_steps, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0, help="Noam lr factor (paper uses 1.0)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default="transformer_best.pt")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    if DEFAULT_WARMUP_STEPS <= 0:
        raise ValueError("DEFAULT_WARMUP_STEPS must be > 0")

    print(f"Using device: {DEVICE}")

    train_limit = None
    dev_limit = None
    test_limit = None
    train_steps = None
    dev_steps = None
    test_steps = None
    num_epochs = args.epochs

    if args.smoke:
        train_limit = DEFAULT_SMOKE_TRAIN_SIZE
        dev_limit = DEFAULT_SMOKE_DEV_SIZE
        test_limit = DEFAULT_SMOKE_TEST_SIZE
        train_steps = DEFAULT_SMOKE_TRAIN_STEPS
        dev_steps = DEFAULT_SMOKE_EVAL_STEPS
        test_steps = DEFAULT_SMOKE_EVAL_STEPS
        num_epochs = 1
        print(
            "Smoke mode enabled: "
            f"train/dev/test sizes={train_limit}/{dev_limit}/{test_limit}, "
            f"train/dev/test steps={train_steps}/{dev_steps}/{test_steps}, "
            f"epochs={num_epochs}"
        )

    train_loader, dev_loader, test_loader = prepare_dataloaders(
        split_seed=SPLIT_SEED,
        train_limit=train_limit,
        dev_limit=dev_limit,
        test_limit=test_limit,
    )
    vocab_size = len(tokenizer)
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        pad_id=PAD_ID,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = Adam(
        model.parameters(),
        lr=1.0,
        betas=(DEFAULT_BETA1, DEFAULT_BETA2),
        eps=DEFAULT_ADAM_EPS,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    lr_scheduler_state = {
        "step_num": 0,
        "model_dim": MODEL_DIM,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "lr_factor": args.lr,
        "last_lr": noam_learning_rate(1, MODEL_DIM, DEFAULT_WARMUP_STEPS) * args.lr,
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    metrics_path = save_dir / DEFAULT_METRICS_NAME
    best_dev_loss = float("inf")
    no_improve_count = 0
    metrics = {"epochs": [], "final": {}}

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=DEVICE,
            stage_name="train",
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            max_steps=train_steps,
            lr_scheduler_state=lr_scheduler_state,
        )
        dev_loss = run_epoch(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=DEVICE,
            stage_name="dev",
            optimizer=None,
            max_steps=dev_steps,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}"
        )
        metrics["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "dev_loss": float(dev_loss),
                "last_lr": float(lr_scheduler_state["last_lr"]),
                "step_num": int(lr_scheduler_state["step_num"]),
            }
        )
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        should_stop = False
        if dev_loss < best_dev_loss - args.min_delta:
            best_dev_loss = dev_loss
            no_improve_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best checkpoint to {save_path}")
        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}/{args.patience}")
            if no_improve_count >= args.patience:
                should_stop = True

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        print(f"Loaded best checkpoint from {save_path}")

    test_loss = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=DEVICE,
        stage_name="test",
        optimizer=None,
        max_steps=test_steps,
    )
    print(f"test_loss={test_loss:.4f}")

    # get_bleu_score reads checkpoint from eval_bleu.CHECKPOINT_PATH.
    if not save_path.exists():
        torch.save(model.state_dict(), save_path)
    eval_bleu.CHECKPOINT_PATH = save_path
    test_bleu = eval_bleu.get_bleu_score(test_loader)
    print(f"test_bleu={test_bleu:.2f}")

    metrics["final"] = {
        "test_loss": float(test_loss),
        "test_bleu": float(test_bleu),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
