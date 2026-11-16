"""Turn parsed battles into per-player token sequences plus targets.

Each battle yields two sequences, one per player perspective. Token 0 is a
START token at tick 0; token k >= 1 is the (k-1)-th event of the battle (either
player, plays and ability activations). The output at token i is a query made
at that token's tick, after seeing tokens 0..i. Its targets describe the
perspective player's *next own card play* after token i: card, tile (own
frame) and delay in ticks (time until next play; "play within 1 s" is the
wait/play decision), plus the battle outcome.
"pre_play" marks queries whose next token is that own play, i.e. the query a
policy makes right before acting; card and tile are scored on those.

Splits (IL_Replay has no dates or player IDs, so held-out sets are by battle):
  test_shard  : every battle in the last HELD_OUT_SHARDS parquet shards, a rough
                proxy for a later collection batch
  test_random : 4% of the remaining battles, by hash of the battle id
  val         : 2% of the remaining battles, by hash
  train       : the rest

Usage: python -m clashai.dataset --parsed data/il_parsed --out data/il_seq [--max-train-battles N]
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os

import numpy as np

from .format import GRID_W, GRID_H, Battle, tile_for

PAD = 0
UNK = 1
START = 2
NO_TILE = GRID_W * GRID_H
SIDE_NONE = 0
SIDE_OWN = 1
SIDE_OPP = 2
KIND_START = 0
KIND_PLAY = 1
KIND_ABILITY = 2
MAX_TOKENS = 256
HELD_OUT_SHARDS = 3
TEST_RANDOM_PCT = 4
VAL_PCT = 2
SPLITS = ("train", "val", "test_random", "test_shard")
TOKEN_FIELDS = ("card", "side", "kind", "tile", "tick", "t_card", "t_tile", "t_delay", "pre_play")


def assign_split(battle_id: str, shard_index: int, n_shards: int) -> str:
    if shard_index >= n_shards - HELD_OUT_SHARDS:
        return "test_shard"
    h = int(hashlib.md5(battle_id.encode()).hexdigest()[:8], 16) % 100
    if h < TEST_RANDOM_PCT:
        return "test_random"
    if h < TEST_RANDOM_PCT + VAL_PCT:
        return "val"
    return "train"


def perspective_rows(battle: Battle, side: str, card_id):
    """Per-token inputs and targets for one player's view of one battle."""
    winner = battle.winner()
    if winner is None:
        win = -1
    elif winner == side:
        win = 1
    else:
        win = 0

    events = battle.events[: MAX_TOKENS - 1]
    n = len(events) + 1
    card = np.zeros(n, dtype=np.int16)
    sidev = np.zeros(n, dtype=np.int8)
    kind = np.zeros(n, dtype=np.int8)
    tile = np.full(n, NO_TILE, dtype=np.int16)
    tick = np.zeros(n, dtype=np.int16)
    card[0] = START
    kind[0] = KIND_START
    for k in range(len(events)):
        e = events[k]
        i = k + 1
        card[i] = card_id(e.card)
        if e.side == side:
            sidev[i] = SIDE_OWN
        else:
            sidev[i] = SIDE_OPP
        if e.kind == "play":
            kind[i] = KIND_PLAY
            tx, ty = tile_for(e, side)
            tile[i] = ty * GRID_W + tx
        else:
            kind[i] = KIND_ABILITY
        tick[i] = e.tick

    # Targets: scan backwards so each position knows its next own play.
    t_card = np.full(n, -1, dtype=np.int16)
    t_tile = np.full(n, -1, dtype=np.int16)
    t_delay = np.full(n, -1, dtype=np.int16)
    pre_play = np.zeros(n, dtype=np.int8)
    next_idx = -1
    for i in range(n - 1, -1, -1):
        if next_idx >= 0:
            t_card[i] = card[next_idx]
            t_tile[i] = tile[next_idx]
            t_delay[i] = tick[next_idx] - tick[i]
            if next_idx == i + 1:
                pre_play[i] = 1
        if sidev[i] == SIDE_OWN and kind[i] == KIND_PLAY:
            next_idx = i

    deck = np.zeros(8, dtype=np.int16)
    own_deck = battle.decks[side]
    for j in range(min(8, len(own_deck))):
        deck[j] = card_id(own_deck[j])
    return {
        "card": card, "side": sidev, "kind": kind, "tile": tile, "tick": tick,
        "t_card": t_card, "t_tile": t_tile, "t_delay": t_delay, "pre_play": pre_play,
        "deck": deck, "win": win,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data/il_parsed")
    ap.add_argument("--out", default="data/il_seq")
    ap.add_argument("--max-train-battles", type=int, default=0, help="0 = all; for smoke runs")
    ap.add_argument("--max-eval-battles", type=int, default=0, help="0 = all; per held-out split")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.parsed, "part-*.jsonl.gz")))
    # Card vocabulary covers every card in every split: a player's own deck is
    # always known, so this leaks nothing about the targets.
    vocab = {"<pad>": PAD, "<unk>": UNK, "<start>": START}

    def card_id(c):
        if c not in vocab:
            vocab[c] = len(vocab)
        return vocab[c]

    cols = {}
    other = {}
    counts = collections.Counter()
    for name in SPLITS:
        cols[name] = {}
        for f in TOKEN_FIELDS:
            cols[name][f] = []
        other[name] = {"deck": [], "win": [], "length": [], "battle_index": []}
    for si in range(len(shards)):
        with gzip.open(shards[si], "rt") as fh:
            for line in fh:
                d = json.loads(line)
                name = assign_split(d["battle_id"], si, len(shards))
                if name == "train" and args.max_train_battles and counts[name] >= args.max_train_battles:
                    continue
                if name != "train" and args.max_eval_battles and counts[name] >= args.max_eval_battles:
                    continue
                b = Battle.from_dict(d)
                for side in ("team", "opponent"):
                    r = perspective_rows(b, side, card_id)
                    for f in TOKEN_FIELDS:
                        cols[name][f].append(r[f])
                    other[name]["deck"].append(r["deck"])
                    other[name]["win"].append(r["win"])
                    other[name]["length"].append(len(r["card"]))
                    other[name]["battle_index"].append(counts[name])
                counts[name] += 1
        print(os.path.basename(shards[si]), dict(counts), flush=True)

    summary = {}
    for name in SPLITS:
        arrays = {}
        for f in TOKEN_FIELDS:
            arrays[f] = np.concatenate(cols[name][f])
        cols[name] = None
        lengths = np.array(other[name]["length"], dtype=np.int32)
        arrays["deck"] = np.stack(other[name]["deck"])
        arrays["win"] = np.array(other[name]["win"], dtype=np.int8)
        arrays["length"] = lengths
        arrays["offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        arrays["battle_index"] = np.array(other[name]["battle_index"], dtype=np.int32)
        np.savez(os.path.join(args.out, f"{name}.npz"), **arrays)
        summary[name] = {"battles": counts[name], "sequences": int(len(lengths)), "tokens": int(lengths.sum()),
                         "truncated_sequences": int((lengths == MAX_TOKENS).sum())}
        print(name, summary[name], flush=True)
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(args.out, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
