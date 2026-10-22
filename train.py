"""Train one Transformer LM configuration on PTB.

Example:
    python train.py --nhead 4 --num_layers 4

Saves the checkpoint with the best validation perplexity to
checkpoints/h{nhead}_l{num_layers}.pt and per-epoch history to
results/h{nhead}_l{num_layers}.json.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from data import load_ptb, make_windows
from model import TransformerLM


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, x, y, batch_size, device):
    """Perplexity over all windows (token-weighted mean NLL)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction="sum").item()
        total_tokens += yb.numel()
    return math.exp(total_loss / total_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--force", action="store_true", help="retrain even if results already exist")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    name = f"h{args.nhead}_l{args.num_layers}"
    ckpt_path = os.path.join(args.out_dir, "checkpoints", f"{name}.pt")
    hist_path = os.path.join(args.out_dir, "results", f"{name}.json")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    if not args.force and os.path.exists(hist_path):
        with open(hist_path) as f:
            if len(json.load(f)["epoch"]) >= args.epochs:
                print(f"[{name}] already trained ({hist_path}); skipping. Use --force to retrain.")
                return

    vocab, streams = load_ptb(args.data_dir)
    windows = {s: make_windows(t, args.context_len) for s, t in streams.items()}
    train_x, train_y = windows["train"]

    config = dict(vocab_size=len(vocab), d_model=args.d_model, nhead=args.nhead,
                  num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                  dropout=args.dropout, context_len=args.context_len)
    model = TransformerLM(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] device={device} params={n_params:,} vocab={len(vocab)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_x) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup = steps_per_epoch  # 1 epoch linear warmup, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup,
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup))))))

    history = dict(config=config, args=vars(args), n_params=n_params, epoch=[],
                   train_ppl=[], val_ppl=[], test_ppl=[], epoch_time=[])
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)
            loss = F.cross_entropy(model(xb).flatten(0, 1), yb.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if device.type == "mps":
            torch.mps.synchronize()
        epoch_time = time.time() - t0

        ppl = {s: evaluate(model, *windows[s], args.batch_size, device)
               for s in ("train", "valid", "test")}
        for key, s in (("train_ppl", "train"), ("val_ppl", "valid"), ("test_ppl", "test")):
            history[key].append(ppl[s])
        history["epoch"].append(epoch)
        history["epoch_time"].append(epoch_time)

        marker = ""
        if ppl["valid"] < best_val:
            best_val = ppl["valid"]
            history["best_epoch"] = epoch
            torch.save(dict(config=config, state_dict=model.state_dict(), epoch=epoch), ckpt_path)
            marker = " *"
        print(f"[{name}] epoch {epoch:2d} | {epoch_time:5.1f}s | train {ppl['train']:7.2f} "
              f"| val {ppl['valid']:7.2f} | test {ppl['test']:7.2f}{marker}", flush=True)

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=1)


if __name__ == "__main__":
    main()
