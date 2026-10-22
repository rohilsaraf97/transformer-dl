"""Evaluate saved checkpoint(s) on PTB train / valid / test.

Examples:
    python evaluate.py checkpoints/h4_l4.pt
    python evaluate.py checkpoints/*.pt
"""
import argparse

import torch

from data import load_ptb, make_windows
from model import TransformerLM
from train import evaluate, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()

    device = get_device()
    _, streams = load_ptb(args.data_dir)
    print(f"{'checkpoint':<28} {'epoch':>5} {'train':>8} {'valid':>8} {'test':>8}")
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device)
        model = TransformerLM(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        ctx = ckpt["config"]["context_len"]
        ppl = [evaluate(model, *make_windows(streams[s], ctx), args.batch_size, device)
               for s in ("train", "valid", "test")]
        print(f"{path:<28} {ckpt['epoch']:>5} " + " ".join(f"{v:8.2f}" for v in ppl))


if __name__ == "__main__":
    main()
