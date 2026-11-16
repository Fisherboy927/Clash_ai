"""Train the warm-up model. Usage: python -m clashai.train --config configs/warmup.json

Checkpoints go to run_dir (gitignored); never commit them.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .model import PlayHistoryTransformer, collate, length_batches, losses, count_params, cosine_lr


def load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    out = {}
    for k in d.files:
        out[k] = d[k]
    return out


@torch.no_grad()
def eval_loss(model, arrays, cfg):
    model.eval()
    totals = {"card": 0.0, "tile": 0.0, "delay": 0.0, "win": 0.0}
    n = 0
    for ids in length_batches(arrays["length"], cfg["max_tokens_per_batch"]):
        batch = collate(arrays, ids)
        out = model(batch)
        _, parts = losses(model, out, batch, cfg["loss_weights"])
        for k in totals:
            totals[k] += parts[k] * len(ids)
        n += len(ids)
    model.train()
    for k in totals:
        totals[k] /= n
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/warmup.json")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["run_dir"], exist_ok=True)

    with open(os.path.join(cfg["data_dir"], "vocab.json")) as f:
        vocab = json.load(f)
    train = load_split(cfg["data_dir"], "train")
    val = load_split(cfg["data_dir"], "val")

    model = PlayHistoryTransformer(len(vocab), **cfg["model"])
    print("params", count_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epoch_batches = []
    for epoch in range(cfg["epochs"]):
        epoch_batches.append(length_batches(train["length"], cfg["max_tokens_per_batch"], rng))
    total_steps = 0
    for batches in epoch_batches:
        total_steps += len(batches)
    step = 0
    best = None
    log = []
    t0 = time.time()
    running = {"card": 0.0, "tile": 0.0, "delay": 0.0, "win": 0.0}
    running_n = 0
    for epoch in range(cfg["epochs"]):
        batches = epoch_batches[epoch]
        for bi in range(len(batches)):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, cfg["warmup_steps"], cfg["lr"])
            batch = collate(train, batches[bi])
            out = model(batch)
            loss, parts = losses(model, out, batch, cfg["loss_weights"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            for k in running:
                running[k] += parts[k]
            running_n += 1
            if step % 200 == 0:
                msg = {}
                for k in running:
                    msg[k] = round(running[k] / running_n, 4)
                running = {"card": 0.0, "tile": 0.0, "delay": 0.0, "win": 0.0}
                running_n = 0
                print(f"epoch {epoch} step {step}/{total_steps} {msg} {time.time() - t0:.0f}s", flush=True)
        v = eval_loss(model, val, cfg)
        v_total = v["card"] + v["tile"] + 0.5 * v["delay"] + 0.5 * v["win"]
        print(f"epoch {epoch} val {v} total {v_total:.4f}", flush=True)
        log.append({"epoch": epoch, "val": v, "val_total": v_total, "elapsed_s": time.time() - t0})
        if best is None or v_total < best:
            best = v_total
            torch.save({"model": model.state_dict(), "config": cfg, "vocab_size": len(vocab), "epoch": epoch},
                       os.path.join(cfg["run_dir"], "best.pt"))
    with open(os.path.join(cfg["run_dir"], "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
