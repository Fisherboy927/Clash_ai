"""Convert VanguardX101/IL_Replay parquet shards into the shared `Battle` format.

The dataset is already parsed from RoyaleAPI replays by FirstLight_CR
(royaleapi-replay-parquet.v1): actions carry 20 Hz ticks and 18 x 32 grid cells
in a frame with the team at the bottom. We read it one shard at a time.

Usage:
    python -m clashai.il_replay --root data/il --out data/il_parsed

Writes data/il_parsed/part-XXXXXX.jsonl.gz (one Battle per line) and
data/il_parsed/parse_report.json (kept and dropped battles by reason).
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq

from .format import Battle, Event

MIN_PLAYS_TOTAL = 12
MIN_PLAYS_PER_SIDE = 5
MAX_TICK = 20 * 60 * 6
VARIANT_SUFFIX = re.compile(r"-(ev\d+|hero)$")
ACTION_COLUMNS = ["replay_tag", "event_index", "kind", "side", "card_key", "replay_tick_20hz",
                  "grid_x", "grid_y", "ability_source_candidates"]


class Drop(Exception):
    pass


def base_card(key: str) -> str:
    """Deck entries name evolution/hero variants (hog-rider-ev1, knight-hero); plays use the base card."""
    return VARIANT_SUFFIX.sub("", key)


def battle_from_rows(replay: dict, actions: list[dict]) -> Battle:
    payload = json.loads(replay["payload_json"])
    decks = {}
    for side in ("team", "opponent"):
        players = payload["battle"][side]["players"]
        if len(players) != 1:
            raise Drop("not_1v1")
        deck = []
        for c in players[0]["deck"]:
            deck.append(base_card(c["card_key"]))
        if len(deck) != 8:
            raise Drop("deck_not_8_cards")
        decks[side] = deck
    crowns = {"team": replay["team_crowns"], "opponent": replay["opponent_crowns"]}
    if crowns["team"] is None or crowns["opponent"] is None:
        raise Drop("missing_crowns")

    events = []
    plays = {"team": 0, "opponent": 0}
    low = {"team": 0, "opponent": 0}
    for a in sorted(actions, key=lambda r: r["event_index"]):
        side = a["side"]
        if side not in plays:
            raise Drop("unknown_side")
        tick = a["replay_tick_20hz"]
        if a["kind"] == "play_card":
            if a["grid_x"] is None or a["grid_y"] is None:
                raise Drop("play_without_coordinates")
            if a["card_key"] not in decks[side]:
                raise Drop("played_card_not_in_deck")
            events.append(Event(tick=tick, side=side, kind="play", card=a["card_key"],
                                tile_x=int(a["grid_x"]), tile_y=int(a["grid_y"])))
            plays[side] += 1
            if a["grid_y"] < 16:
                low[side] += 1
        elif a["kind"] == "activate_ability":
            cands = a["ability_source_candidates"] or []
            if len(cands) == 1:
                card = base_card(cands[0])
            else:
                card = "ability"
            events.append(Event(tick=tick, side=side, kind="ability", card=card))
        else:
            raise Drop("unknown_action_kind")
    events.sort(key=lambda e: (e.tick, e.side, e.kind))

    if plays["team"] + plays["opponent"] < MIN_PLAYS_TOTAL:
        raise Drop("too_few_plays")
    if plays["team"] < MIN_PLAYS_PER_SIDE or plays["opponent"] < MIN_PLAYS_PER_SIDE:
        raise Drop("one_side_too_few_plays")
    if events[-1].tick > MAX_TICK:
        raise Drop("tick_beyond_6_minutes")
    # Players place most cards on their own half; a team that mostly plays on
    # the top half means the board frame was not normalised for this replay.
    if low["team"] / plays["team"] < low["opponent"] / plays["opponent"]:
        raise Drop("suspect_board_orientation")

    return Battle(
        battle_id=replay["replay_tag"],
        source="IL_Replay",
        captured_at=None,
        battle_time=None,
        game_mode=f"{replay['battle_type']}/{replay['game_mode']}",
        decks=decks,
        crowns=crowns,
        events=events,
    )


def deck_elixir_rows(replay: dict):
    """(base deck, average elixir) per player, used to solve per-card costs."""
    payload = json.loads(replay["payload_json"])
    rows = []
    for side in ("team", "opponent"):
        for p in payload["battle"][side]["players"]:
            avg = p.get("average_elixir")
            if avg is None or len(p["deck"]) != 8:
                continue
            deck = []
            for c in p["deck"]:
                deck.append(base_card(c["card_key"]))
            rows.append((deck, float(avg)))
    return rows


def solve_card_costs(rows):
    """Least squares: 8 * average_elixir = sum of the deck's card costs."""
    cards = {}
    for deck, _ in rows:
        for c in deck:
            if c not in cards:
                cards[c] = len(cards)
    A = np.zeros((len(rows), len(cards)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.float32)
    for i in range(len(rows)):
        deck, avg = rows[i]
        for c in deck:
            A[i, cards[c]] += 1
        y[i] = 8 * avg
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    costs = {}
    for c in cards:
        costs[c] = int(round(float(x[cards[c]])))
    return costs


def convert_shard(replay_path: str, action_path: str, out_path: str, drops, modes, totals, cost_rows):
    replays = pq.read_table(replay_path, columns=["replay_tag", "battle_type", "game_mode", "team_crowns",
                                                  "opponent_crowns", "payload_json"]).to_pylist()
    actions = pq.read_table(action_path, columns=ACTION_COLUMNS).to_pylist()
    by_tag = collections.defaultdict(list)
    for a in actions:
        by_tag[a["replay_tag"]].append(a)
    del actions
    with gzip.open(out_path, "wt") as f:
        for r in replays:
            totals["replays"] += 1
            if len(cost_rows) < 200_000:
                cost_rows.extend(deck_elixir_rows(r))
            acts = by_tag.pop(r["replay_tag"], [])
            if not acts:
                drops["no_actions"] += 1
                continue
            try:
                b = battle_from_rows(r, acts)
            except Drop as d:
                drops[str(d)] += 1
                continue
            totals["kept_battles"] += 1
            modes[b.game_mode] += 1
            for e in b.events:
                if e.kind == "play":
                    totals["kept_card_plays"] += 1
                else:
                    totals["kept_ability_events"] += 1
            totals["outcome_" + (b.winner() or "draw")] += 1
            f.write(json.dumps(b.to_dict()) + "\n")
    totals["actions_without_replay_row"] += sum(len(v) for v in by_tag.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/il")
    ap.add_argument("--out", default="data/il_parsed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    drops = collections.Counter()
    modes = collections.Counter()
    totals = collections.Counter()
    cost_rows = []
    replay_files = sorted(glob.glob(os.path.join(args.root, "replays", "part-*.parquet")))
    for rp in replay_files:
        name = os.path.basename(rp)
        ap_path = os.path.join(args.root, "actions", name)
        out_path = os.path.join(args.out, name.replace(".parquet", ".jsonl.gz"))
        convert_shard(rp, ap_path, out_path, drops, modes, totals, cost_rows)
        print(name, dict(totals), flush=True)
    with open(os.path.join(args.out, "card_costs.json"), "w") as f:
        json.dump(solve_card_costs(cost_rows), f, indent=1, sort_keys=True)
    report = {"shards": len(replay_files), **dict(totals), "game_modes": dict(modes.most_common()),
              "dropped": dict(drops.most_common())}
    with open(os.path.join(args.out, "parse_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
