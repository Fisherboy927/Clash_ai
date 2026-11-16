"""Turn parsed battles into per-player token sequences plus targets.

Each battle yields two sequences, one per player perspective. Token 0 is a
START token at tick 0; token k >= 1 is the (k-1)-th event of the battle (either
player, plays and ability activations). The output at token i is a query made
at that token's tick, after seeing tokens 0..i. Its targets describe the
perspective player's *next own card play* after token i:
  card, tile (own frame), delay in ticks, and whether it lands within 1 s
  (the wait/play decision on a 1 s step), plus the battle outcome.
"pre_play" marks queries whose next token is that own play, i.e. the query a
policy makes right before acting; card and tile are scored on those.

Usage: python -m clashai.dataset --parsed data/parsed --out data/seq
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random

import numpy as np

from .format import GRID_W, GRID_H, TICKS_PER_SECOND, Battle, tile_for

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
TEST_TIME_DAY = "2026-08-11"
TEST_PLAYER_PCT = 8
VAL_PLAYER_PCT = 3
PLAY_SOON_TICKS = TICKS_PER_SECOND
MAX_TOKENS = 256


def player_bucket(tag) -> int:
    if not tag:
        return 99
    return int(hashlib.md5(tag.encode()).hexdigest()[:8], 16) % 100


def assign_split(b: dict, rng: random.Random) -> str:
    """test_time: captured on the last day. test_player / val: earlier battles
    with a held-out player (no battle of theirs is in train); val players are a
    separate bucket so model selection never sees test players. The rest is
    split at random by battle into train / test_random."""
    cap = b.get("captured_at")
    if cap and cap[:10] >= TEST_TIME_DAY:
        return "test_time"
    players = b.get("players") or {}
    buckets = [player_bucket(players.get("team")), player_bucket(players.get("opponent"))]
    for bk in buckets:
        if bk < TEST_PLAYER_PCT:
            return "test_player"
    for bk in buckets:
        if bk < TEST_PLAYER_PCT + VAL_PLAYER_PCT:
            return "val"
    if rng.random() < 0.05:
        return "test_random"
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
    tick = np.zeros(n, dtype=np.int32)
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
    t_delay = np.full(n, -1, dtype=np.int32)
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
    ap.add_argument("--parsed", default="data/parsed")
    ap.add_argument("--out", default="data/seq")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    battles = []
    splits = []
    with gzip.open(os.path.join(args.parsed, "battles.jsonl.gz"), "rt") as f:
        for line in f:
            d = json.loads(line)
            splits.append(assign_split(d, rng))
            battles.append(d)

    vocab = {"<pad>": PAD, "<unk>": UNK, "<start>": START}
    for k in range(len(battles)):
        if splits[k] != "train":
            continue
        for e in battles[k]["events"]:
            if e["card"] not in vocab:
                vocab[e["card"]] = len(vocab)
        for side in ("team", "opponent"):
            for c in battles[k]["decks"][side]:
                if c not in vocab:
                    vocab[c] = len(vocab)

    def card_id(c):
        return vocab.get(c, UNK)

    split_ids = {}
    train_players = set()
    for k in range(len(battles)):
        if splits[k] == "train":
            for tag in (battles[k].get("players") or {}).values():
                if tag:
                    train_players.add(tag)
    overlap = {}
    for name in ("val", "test_random", "test_player", "test_time"):
        n = 0
        seen = 0
        for k in range(len(battles)):
            if splits[k] != name:
                continue
            n += 1
            tags = (battles[k].get("players") or {}).values()
            for tag in tags:
                if tag in train_players:
                    seen += 1
                    break
        overlap[name] = {"battles": n, "with_a_player_seen_in_train": seen}
    print("player overlap with train", overlap)

    for name in ("train", "val", "test_random", "test_player", "test_time"):
        cols = {"card": [], "side": [], "kind": [], "tile": [], "tick": [],
                "t_card": [], "t_tile": [], "t_delay": [], "pre_play": []}
        decks = []
        wins = []
        lengths = []
        battle_index = []
        player_seen = []
        ids = []
        for k in range(len(battles)):
            if splits[k] != name:
                continue
            b = Battle.from_dict(battles[k])
            ids.append(b.battle_id)
            for side in ("team", "opponent"):
                r = perspective_rows(b, side, card_id)
                for c in cols:
                    cols[c].append(r[c])
                player_seen.append(1 if b.players.get(side) in train_players else 0)
                decks.append(r["deck"])
                wins.append(r["win"])
                lengths.append(len(r["card"]))
                battle_index.append(len(ids) - 1)
        arrays = {}
        for c in cols:
            arrays[c] = np.concatenate(cols[c])
        arrays["deck"] = np.stack(decks)
        arrays["win"] = np.array(wins, dtype=np.int8)
        arrays["length"] = np.array(lengths, dtype=np.int32)
        arrays["offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        arrays["battle_index"] = np.array(battle_index, dtype=np.int32)
        arrays["player_seen_in_train"] = np.array(player_seen, dtype=np.int8)
        np.savez(os.path.join(args.out, f"{name}.npz"), **arrays)
        split_ids[name] = ids
        print(name, "battles", len(ids), "sequences", len(lengths), "tokens", int(sum(lengths)))

    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(split_ids, f)
    summary = {"player_overlap_with_train": overlap, "battles_per_split": collections.Counter(splits)}
    with open(os.path.join(args.out, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
