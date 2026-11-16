"""Download VanguardX101/IL_Replay from Hugging Face into data/il and verify checksums.

Usage: python -m clashai.download [--root data/il] [--max-shards N]

The data stays in the gitignored data/ directory; never commit or redistribute it.
"""

from __future__ import annotations

import argparse
import hashlib
import os

from huggingface_hub import HfApi, hf_hub_download

REPO = "VanguardX101/IL_Replay"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/il")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all; download only the first N shards")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    shard_names = []
    for f in files:
        if f.startswith("replays/part-") and f.endswith(".parquet"):
            shard_names.append(os.path.basename(f))
    shard_names.sort()
    if args.max_shards:
        shard_names = shard_names[: args.max_shards]

    wanted = ["manifest.json", "SHA256SUMS"]
    for name in shard_names:
        wanted.append("replays/" + name)
        wanted.append("actions/" + name)
    for f in wanted:
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.root)
        print("downloaded", f, flush=True)

    expected = {}
    with open(os.path.join(args.root, "SHA256SUMS")) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                expected[parts[1]] = parts[0]
    bad = []
    for f in wanted:
        if f in expected and sha256(os.path.join(args.root, f)) != expected[f]:
            bad.append(f)
    if bad:
        raise SystemExit(f"checksum mismatch: {bad}")
    print(f"ok: {len(shard_names)} shards verified", flush=True)


if __name__ == "__main__":
    main()
