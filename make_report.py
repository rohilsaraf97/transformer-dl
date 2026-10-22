"""Build plots and the summary table from results/*.json (written by train.py).

Outputs:
    plots/h{H}_l{L}.png   train/test perplexity per epoch, one per config
    plots/all_configs.png 3x3 grid of the same curves
    results/summary.md    table of train/val/test perplexity at the best-val epoch
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_COLOR, TEST_COLOR, VAL_COLOR = "#1f77b4", "#d62728", "#7f7f7f"


def load_results(results_dir="results"):
    runs = []
    for path in glob.glob(os.path.join(results_dir, "h*_l*.json")):
        with open(path) as f:
            h = json.load(f)
        m = re.search(r"h(\d+)_l(\d+)", path)
        h["nhead"], h["num_layers"] = int(m.group(1)), int(m.group(2))
        runs.append(h)
    return sorted(runs, key=lambda r: (r["nhead"], r["num_layers"]))


def plot_run(ax, r):
    ax.plot(r["epoch"], r["train_ppl"], color=TRAIN_COLOR, marker="o", ms=3, label="Train")
    ax.plot(r["epoch"], r["test_ppl"], color=TEST_COLOR, marker="o", ms=3, label="Test")
    ax.plot(r["epoch"], r["val_ppl"], color=VAL_COLOR, ls="--", lw=1, label="Validation")
    ax.axvline(r["best_epoch"], color="black", ls=":", lw=1, label=f"Best val (epoch {r['best_epoch']})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def title(r):
    return f"PTB Transformer LM: {r['nhead']} heads, {r['num_layers']} layers"


def main():
    runs = load_results()
    os.makedirs("plots", exist_ok=True)

    for r in runs:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_run(ax, r)
        ax.set_title(title(r))
        fig.tight_layout()
        fig.savefig(f"plots/h{r['nhead']}_l{r['num_layers']}.png", dpi=150)
        plt.close(fig)

    heads = sorted({r["nhead"] for r in runs})
    layers = sorted({r["num_layers"] for r in runs})
    fig, axes = plt.subplots(len(heads), len(layers), figsize=(5 * len(layers), 3.6 * len(heads)),
                             sharey=True, squeeze=False)
    for r in runs:
        ax = axes[heads.index(r["nhead"])][layers.index(r["num_layers"])]
        plot_run(ax, r)
        ax.set_title(f"{r['nhead']} heads, {r['num_layers']} layers")
    fig.suptitle("Train / test perplexity per epoch for all configurations", fontsize=14)
    fig.tight_layout()
    fig.savefig("plots/all_configs.png", dpi=120)
    plt.close(fig)

    lines = ["| Heads | Layers | Params | Best epoch | Train PPL | Val PPL | Test PPL | Sec/epoch | Total train time (s) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        i = r["epoch"].index(r["best_epoch"])
        times = r["epoch_time"]
        lines.append(f"| {r['nhead']} | {r['num_layers']} | {r['n_params'] / 1e6:.2f}M | {r['best_epoch']} "
                     f"| {r['train_ppl'][i]:.2f} | {r['val_ppl'][i]:.2f} | {r['test_ppl'][i]:.2f} "
                     f"| {sum(times) / len(times):.1f} | {sum(times):.0f} |")
    best = min(runs, key=lambda r: min(r["val_ppl"]))
    lines.append("")
    lines.append(f"Best configuration by validation perplexity: {best['nhead']} heads, "
                 f"{best['num_layers']} layers (val {min(best['val_ppl']):.2f}, "
                 f"test {best['test_ppl'][best['epoch'].index(best['best_epoch'])]:.2f}).")
    table = "\n".join(lines)
    with open("results/summary.md", "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
