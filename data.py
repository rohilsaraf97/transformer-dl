"""Penn Tree Bank (word-level) loading.

torchtext's dataset loaders are deprecated, so we download the same raw files
(Mikolov's preprocessed PTB, 10k vocab, already contains <unk>) from GitHub.
"""
import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt"
SPLITS = ("train", "valid", "test")
EOS = "<eos>"


def download(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for split in SPLITS:
        path = os.path.join(data_dir, f"ptb.{split}.txt")
        if not os.path.exists(path):
            print(f"Downloading {URL.format(split)}")
            urllib.request.urlretrieve(URL.format(split), path)


def read_words(path):
    # Each line is a sentence; mark sentence boundaries with <eos>.
    with open(path) as f:
        return [w for line in f for w in line.split() + [EOS]]


class Vocab:
    def __init__(self, words):
        self.itos = sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, words):
        return torch.tensor([self.stoi[w] for w in words], dtype=torch.long)


def load_ptb(data_dir="data"):
    """Returns (vocab, {split: 1-D LongTensor of token ids})."""
    download(data_dir)
    words = {s: read_words(os.path.join(data_dir, f"ptb.{s}.txt")) for s in SPLITS}
    vocab = Vocab(words["train"])  # PTB valid/test contain no OOV words w.r.t. train
    return vocab, {s: vocab.encode(w) for s, w in words.items()}


def make_windows(stream, context_len):
    """Cut a token stream into non-overlapping (input, target) windows.

    Returns two tensors of shape (num_windows, context_len), where target is
    input shifted by one token.
    """
    n = (len(stream) - 1) // context_len
    x = stream[: n * context_len].view(n, context_len)
    y = stream[1 : n * context_len + 1].view(n, context_len)
    return x, y
