"""Parse RoyaleAPI replay captures (Cochon123/clash-royale-replays) into `Battle`s.

Usage:
    python -m clashai.parse_royaleapi --snapshot data/hf/snap/data \
        --legacy data/hf/cleaned-legacy --out data/parsed

Writes data/parsed/battles.jsonl.gz (one Battle per line) and
data/parsed/parse_report.json (counts of kept and dropped battles by reason).
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
from multiprocessing import Pool
from urllib.parse import parse_qs, urlparse

from .format import Battle, Event, arena_to_tile

SIDE_MAP = {"t": "team", "blue": "team", "o": "opponent", "red": "opponent"}
TAG_RE = re.compile(r"<(div|img)\b([^>]*)>", re.S)
ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')
ABILITY_SLUG = re.compile(r"ability-([a-z0-9-]+)\.png")
MIN_PLAYS_PER_SIDE = 5
MIN_PLAYS_TOTAL = 12


class Drop(Exception):
    pass


def _int(v):
    if v in (None, "", "None", "null"):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _attrs(tag_body: str) -> dict:
    out = {}
    for m in ATTR_RE.finditer(tag_body):
        out[m.group(1)] = m.group(2)
    return out


def parse_html(html: str, request_url: str | None):
    """Returns (battle_id, decks, crowns, tags, events) or raises Drop(reason)."""
    if "Just a moment..." in html or "challenge-platform" in html:
        raise Drop("cloudflare_challenge")
    m = re.search(r'class="battle_replay"[^>]*data-tag="([A-Z0-9]+)"', html)
    if not m:
        raise Drop("no_battle_replay_div")
    battle_id = m.group(1)

    decks = {"team": [], "opponent": []}
    markers = []
    abilities = []
    stats_start = html.find('class="replay_stats"')
    for tm in TAG_RE.finditer(html):
        kind = tm.group(1)
        a = _attrs(tm.group(2))
        cls = a.get("class", "")
        if kind == "div" and "marker" in cls.split() and "data-t" in a and "data-s" in a:
            markers.append(a)
        elif kind == "img" and "replay_card" in cls.split():
            side = SIDE_MAP.get(a.get("data-s", "").lower())
            if side is None:
                continue
            if "data-t" in a:
                if a.get("data-ability") == "1":
                    abilities.append(a)
            elif stats_start >= 0 and tm.start() > stats_start:
                card = a.get("data-card")
                if card and card != "_invalid" and card not in decks[side]:
                    decks[side].append(card)

    events = []
    for a in markers:
        side = SIDE_MAP.get(a.get("data-s", "").lower())
        tick = _int(a.get("data-t"))
        card = a.get("data-c")
        x = _int(a.get("data-x"))
        y = _int(a.get("data-y"))
        if side is None or tick is None or not card:
            continue
        if card == "_invalid" or x is None or y is None:
            continue
        tx, ty = arena_to_tile(x, y)
        events.append(Event(tick=tick, side=side, kind="play", card=card, tile_x=tx, tile_y=ty))
    for a in abilities:
        side = SIDE_MAP.get(a.get("data-s", "").lower())
        tick = _int(a.get("data-t"))
        if side is None or tick is None:
            continue
        sm = ABILITY_SLUG.search(a.get("src", ""))
        card = sm.group(1) if sm else "ability"
        events.append(Event(tick=tick, side=side, kind="ability", card=card))
    events.sort(key=lambda e: (e.tick, e.side, e.kind))

    query = {}
    if request_url:
        query = parse_qs(urlparse(request_url).query)
    if not query:
        for href in re.findall(r'href="([^"]*replay\?[^"]*)"', html):
            query = parse_qs(urlparse(href.replace("&amp;", "&")).query)
            if query.get("tag"):
                break
    tags = {}
    tags["team"] = [t for t in (query.get("team_tags") or [""])[0].split(",") if t]
    tags["opponent"] = [t for t in (query.get("opponent_tags") or [""])[0].split(",") if t]
    crowns = {}
    crowns["team"] = _int((query.get("team_crowns") or [None])[0])
    crowns["opponent"] = _int((query.get("opponent_crowns") or [None])[0])
    return battle_id, decks, crowns, tags, events


def validate(decks, crowns, tags, events):
    if len(tags["team"]) > 1 or len(tags["opponent"]) > 1:
        raise Drop("not_1v1")
    if crowns["team"] is None or crowns["opponent"] is None:
        raise Drop("missing_crowns")
    plays = collections.Counter()
    for e in events:
        if e.kind == "play":
            plays[e.side] += 1
            if e.card not in decks[e.side]:
                raise Drop("played_card_not_in_deck")
    if plays["team"] + plays["opponent"] < MIN_PLAYS_TOTAL:
        raise Drop("too_few_plays")
    if plays["team"] < MIN_PLAYS_PER_SIDE or plays["opponent"] < MIN_PLAYS_PER_SIDE:
        raise Drop("one_side_too_few_plays")
    for side in ("team", "opponent"):
        if len(decks[side]) > 8:
            raise Drop("deck_over_8_cards")
    if events and events[-1].tick > 20 * 60 * 6:
        raise Drop("tick_beyond_6_minutes")


def normalize_orientation(events):
    """RoyaleAPI draws the team at the bottom in some replays and at the top in
    others. Players place most cards on their own half, so compare the share of
    each side's plays in the lower half; rotate 180 degrees if the team is on top."""
    low = {"team": 0, "opponent": 0}
    n = {"team": 0, "opponent": 0}
    for e in events:
        if e.kind != "play":
            continue
        n[e.side] += 1
        if e.tile_y < 16:
            low[e.side] += 1
    if n["team"] == 0 or n["opponent"] == 0:
        raise Drop("missing_side_actions")
    gap = low["team"] / n["team"] - low["opponent"] / n["opponent"]
    if abs(gap) < 0.3:
        raise Drop("ambiguous_board_orientation")
    if gap < 0:
        for e in events:
            if e.tile_x is not None:
                e.tile_x = 17 - e.tile_x
                e.tile_y = 31 - e.tile_y
        return "rotated"
    return "as_is"


def player_tags(tags):
    out = {}
    for side in ("team", "opponent"):
        if tags.get(side):
            out[side] = tags[side][0].lstrip("#")
        else:
            out[side] = None
    return out


def parse_file(path: str):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return ("drop", "invalid_json", None)
        payload = obj.get("payload") or {}
        html = payload.get("html") if isinstance(payload, dict) else None
        if not isinstance(html, str) or not html.strip():
            return ("drop", "missing_html", None)
        battle_id, decks, crowns, tags, events = parse_html(html, obj.get("request_url"))
        validate(decks, crowns, tags, events)
        orientation = normalize_orientation(events)
        b = Battle(
            battle_id=battle_id,
            source="snapshot-2026-11-01",
            captured_at=obj.get("captured_at"),
            battle_time=None,
            game_mode=None,
            decks=decks,
            crowns=crowns,
            events=events,
            players=player_tags(tags),
        )
        d = b.to_dict()
        d["_orientation"] = orientation
        return ("ok", None, d)
    except Drop as d:
        return ("drop", str(d), None)


def load_legacy(legacy_dir: str):
    """cleaned-legacy is already parsed (Dec 2025); convert to Battle dicts."""
    matches = {}
    with open(os.path.join(legacy_dir, "matches.jsonl")) as f:
        for line in f:
            m = json.loads(line)
            matches[m["battle_id"]] = m
    events = collections.defaultdict(list)
    with open(os.path.join(legacy_dir, "events.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            events[e["battle_id"]].append(e)
    out = []
    for bid, m in matches.items():
        evs = []
        for e in events[bid]:
            if e["event_type"] == "card_play":
                tx, ty = arena_to_tile(e["x"], e["y"])
                evs.append(Event(tick=e["ticks"], side=e["side"], kind="play", card=e["card"], tile_x=tx, tile_y=ty))
            else:
                evs.append(Event(tick=e["ticks"], side=e["side"], kind="ability", card=e.get("ability_card") or "ability"))
        evs.sort(key=lambda e: (e.tick, e.side, e.kind))
        tags = m.get("participant_tags") or {"team": [], "opponent": []}
        out.append((bid, m["decks"], m["crowns"], tags, evs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data/hf/snap/data")
    ap.add_argument("--legacy", default="data/hf/cleaned-legacy")
    ap.add_argument("--out", default="data/parsed")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.snapshot, "raw", "*.json")))
    drops = collections.Counter()
    orientations = collections.Counter()
    kept = {}
    with Pool(args.workers) as pool:
        for status, reason, b in pool.imap_unordered(parse_file, files, chunksize=64):
            if status == "drop":
                drops["snapshot:" + reason] += 1
                continue
            if b["battle_id"] in kept:
                drops["snapshot:duplicate_battle_id"] += 1
                continue
            orientations[b.pop("_orientation")] += 1
            kept[b["battle_id"]] = b

    legacy_rows = load_legacy(args.legacy)
    legacy_kept = 0
    for bid, decks, crowns, tags, evs in legacy_rows:
        if bid in kept:
            drops["legacy:duplicate_of_snapshot"] += 1
            continue
        try:
            validate(decks, crowns, tags, evs)
            orientations[normalize_orientation(evs)] += 1
        except Drop as d:
            drops["legacy:" + str(d)] += 1
            continue
        b = Battle(bid, "cleaned-legacy", None, None, None, decks, crowns, evs, player_tags(tags))
        kept[bid] = b.to_dict()
        legacy_kept += 1

    out_path = os.path.join(args.out, "battles.jsonl.gz")
    n_plays = 0
    n_abil = 0
    winners = collections.Counter()
    with gzip.open(out_path, "wt") as f:
        for bid in sorted(kept):
            b = kept[bid]
            for e in b["events"]:
                if e["kind"] == "play":
                    n_plays += 1
                else:
                    n_abil += 1
            winners[Battle.from_dict(b).winner() or "draw"] += 1
            f.write(json.dumps(b) + "\n")

    report = {
        "snapshot_files": len(files),
        "legacy_battles_in_source": len(legacy_rows),
        "kept_battles": len(kept),
        "kept_from_snapshot": len(kept) - legacy_kept,
        "kept_from_legacy": legacy_kept,
        "kept_card_plays": n_plays,
        "kept_ability_events": n_abil,
        "outcomes": dict(winners),
        "board_orientation": dict(orientations),
        "dropped": dict(drops.most_common()),
    }
    with open(os.path.join(args.out, "parse_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
