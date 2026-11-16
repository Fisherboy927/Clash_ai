"""Outputs at position i must not depend on tokens after i, in train or eval mode.

Run: python -m pytest tests  (or python tests/test_causality.py)
"""

import torch

from clashai.model import PlayHistoryTransformer


def make_batch(B=3, T=30):
    g = torch.Generator().manual_seed(0)
    lengths = [T, T - 7, T - 15]
    pad = torch.ones(B, T, dtype=torch.bool)
    for b in range(B):
        pad[b, :lengths[b]] = False
    return {
        "card": torch.randint(3, 50, (B, T), generator=g),
        "side": torch.randint(0, 3, (B, T), generator=g),
        "kind": torch.randint(0, 3, (B, T), generator=g),
        "tile": torch.randint(0, 577, (B, T), generator=g),
        "tick": torch.arange(T)[None].repeat(B, 1) * 30,
        "deck": torch.randint(3, 50, (B, 8), generator=g),
        "pad": pad,
    }


def test_no_future_leak():
    torch.manual_seed(0)
    model = PlayHistoryTransformer(60, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0)
    batch = make_batch()
    cut = 10
    changed = {}
    for k in batch:
        changed[k] = batch[k].clone()
    changed["card"][:, cut:] = torch.randint(3, 50, changed["card"][:, cut:].shape)
    changed["tile"][:, cut:] = torch.randint(0, 577, changed["tile"][:, cut:].shape)
    for mode in ("train", "eval"):
        getattr(model, mode)()
        with torch.no_grad():
            a = model(batch)
            b = model(changed)
        for head in ("card", "delay", "win"):
            diff = (a[head][:, :cut] - b[head][:, :cut]).abs().max().item()
            assert diff < 1e-5, f"{mode} mode leaks future into {head}: {diff}"


if __name__ == "__main__":
    test_no_future_leak()
    print("ok")
