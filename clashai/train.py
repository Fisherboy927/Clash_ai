"""Train the warm-up model.

Usage: python -m clashai.train --config configs/warmup_gpu.json [--device cuda|mps|cpu]

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

LOSS_KEYS = ("card", "tile", "delay", "win")


def load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    out = {}
    for k in d.files:
        out[k] = d[k]
    return out


def pick_device(requested=None):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device, amp):
    """bf16 on CUDA Ampere+ (4090, L4, A100...), fp16 on older CUDA, off elsewhere.
    CPU bf16 autocast is off because oneDNN caches kernels per batch shape and
    memory grows without bound with length-bucketed batches."""
    if not amp:
        return None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


class BatchDataset(torch.utils.data.Dataset):
    """One item = one pre-built length-bucketed batch, collated in a worker."""

    def __init__(self, arrays, batches):
        self.arrays = arrays
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        batch = collate(self.arrays, self.batches[i])
        batch["n_seq"] = torch.tensor(len(self.batches[i]))
        return batch


def loader(arrays, batches, workers, device):
    return torch.utils.data.DataLoader(BatchDataset(arrays, batches), batch_size=None, shuffle=False,
                                       num_workers=workers, pin_memory=device.type == "cuda",
                                       persistent_workers=False)


def to_device(batch, device):
    out = {}
    for k in batch:
        out[k] = batch[k].to(device, non_blocking=True)
    return out


@torch.no_grad()
def eval_loss(model, arrays, cfg, device, amp_dtype):
    model.eval()
    totals = {}
    for k in LOSS_KEYS:
        totals[k] = 0.0
    n = 0
    batches = length_batches(arrays["length"], cfg["max_tokens_per_batch"])
    for batch in loader(arrays, batches, cfg["loader_workers"], device):
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
            _, parts = losses(model, out, batch, cfg["loss_weights"])
        k_seq = int(batch["n_seq"])
        for k in totals:
            totals[k] += parts[k] * k_seq
        n += k_seq
    model.train()
    for k in totals:
        totals[k] /= n
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/warmup_gpu.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--loader-workers", type=int, default=None, help="override the config value")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    if args.loader_workers is not None:
        cfg["loader_workers"] = args.loader_workers
    torch.manual_seed(cfg["seed"])
    if cfg["loader_workers"] > 0:
        # Container /dev/shm is often small (RunPod, Docker defaults).
        torch.multiprocessing.set_sharing_strategy("file_system")
    if cfg.get("threads"):
        torch.set_num_threads(cfg["threads"])
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["run_dir"], exist_ok=True)
    device = pick_device(args.device)
    amp_dtype = autocast_dtype(device, cfg.get("amp", True))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device {device} amp {amp_dtype}", flush=True)

    with open(os.path.join(cfg["data_dir"], "vocab.json")) as f:
        vocab = json.load(f)
    train = load_split(cfg["data_dir"], "train")
    val = load_split(cfg["data_dir"], "val")
    print(f"train sequences {len(train['length'])} tokens {int(train['length'].sum())}", flush=True)

    model = PlayHistoryTransformer(len(vocab), **cfg["model"]).to(device)
    print("params", count_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    epoch_batches = []
    for epoch in range(cfg["epochs"]):
        epoch_batches.append(length_batches(train["length"], cfg["max_tokens_per_batch"], rng))
    total_steps = 0
    for batches in epoch_batches:
        total_steps += len(batches)
    max_steps = cfg.get("max_steps") or total_steps
    total_steps = min(total_steps, max_steps)
    print(f"steps {total_steps}", flush=True)

    step = 0
    best = None
    log = []
    t0 = time.time()
    running = {}
    for k in LOSS_KEYS:
        running[k] = 0.0
    running_n = 0
    tokens_seen = 0
    max_seconds = cfg.get("max_hours", 0) * 3600
    out_of_time = False
    for epoch in range(cfg["epochs"]):
        if step >= max_steps or out_of_time:
            break
        for batch in loader(train, epoch_batches[epoch], cfg["loader_workers"], device):
            if step >= max_steps:
                break
            if max_seconds and time.time() - t0 > max_seconds:
                print(f"stopping: max_hours {cfg['max_hours']} reached at step {step}", flush=True)
                out_of_time = True
                break
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, cfg["warmup_steps"], cfg["lr"])
            batch = to_device(batch, device)
            tokens_seen += int((~batch["pad"]).sum())
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(batch)
                loss, parts = losses(model, out, batch, cfg["loss_weights"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            for k in running:
                running[k] += parts[k]
            running_n += 1
            if step % cfg.get("log_every", 200) == 0:
                msg = {}
                for k in running:
                    msg[k] = round(running[k] / running_n, 4)
                    running[k] = 0.0
                running_n = 0
                elapsed = time.time() - t0
                eta = elapsed / step * (total_steps - step)
                print(f"epoch {epoch} step {step}/{total_steps} {msg} {tokens_seen / elapsed:.0f} tok/s "
                      f"elapsed {elapsed / 60:.1f} min eta {eta / 60:.1f} min", flush=True)
        v = eval_loss(model, val, cfg, device, amp_dtype)
        w = cfg["loss_weights"]
        v_total = 0.0
        for k in LOSS_KEYS:
            v_total += w[k] * v[k]
        print(f"epoch {epoch} val {v} total {v_total:.4f}", flush=True)
        log.append({"epoch": epoch, "step": step, "val": v, "val_total": v_total, "elapsed_s": time.time() - t0,
                    "train_tokens_per_s": tokens_seen / (time.time() - t0)})
        if best is None or v_total < best:
            best = v_total
            torch.save({"model": model.state_dict(), "config": cfg, "vocab_size": len(vocab), "epoch": epoch},
                       os.path.join(cfg["run_dir"], "best.pt"))
        with open(os.path.join(cfg["run_dir"], "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
