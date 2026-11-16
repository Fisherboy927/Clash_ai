"""Shared battle format used by every data source (replays now, video labels later).

Time base: integer ticks at 20 ticks/s, tick 0 = battle start.
Board: fixed 18 x 32 tile grid. Tiles are stored in the *player's own frame*:
tile_x 0..17 left to right, tile_y 0 = own back row, 31 = opponent back row.
The opponent's plays are rotated 180 degrees so both players share one frame.

An action is (kind, card, tile, tick) with kind in {"wait", "play", "ability"}.
Replays only contain "play" and "ability" events; "wait" is implicit between
them and is materialised by `to_decision_steps` for tick-level policies.

Board-state inputs (unit positions, HP, projectiles, elixir) are optional.
Each event can carry a `board` vector; when absent the model sees it masked,
so video-derived board state can be filled in later without a format change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

TICKS_PER_SECOND = 20
GRID_W = 18
GRID_H = 32
ARENA_UNITS_PER_TILE = 1000

SIDES = ("team", "opponent")


@dataclass
class Event:
    tick: int
    side: str  # "team" or "opponent" (RoyaleAPI blue / red)
    kind: str  # "play" or "ability"
    card: str  # card slug; "ability" events use the hero/champion slug or "ability"
    tile_x: int | None = None  # in team (blue) frame; use `tile_for` for a side's frame
    tile_y: int | None = None
    board: list[float] | None = None


@dataclass
class Battle:
    battle_id: str
    source: str  # "snapshot-2026-11-01" or "cleaned-legacy"
    captured_at: str | None  # ISO time the page was captured (proxy for battle time)
    battle_time: str | None  # official battle time when metadata matched
    game_mode: str | None
    decks: dict[str, list[str]]
    crowns: dict[str, int | None]
    events: list[Event] = field(default_factory=list)
    players: dict[str, str | None] = field(default_factory=dict)  # player tags; used only for splitting

    def winner(self) -> str | None:
        t = self.crowns.get("team")
        o = self.crowns.get("opponent")
        if t is None or o is None or t == o:
            return None
        if t > o:
            return "team"
        return "opponent"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Battle":
        events = []
        for e in d["events"]:
            events.append(Event(**e))
        d = dict(d)
        d["events"] = events
        return Battle(**d)


def arena_to_tile(x: int, y: int) -> tuple[int, int]:
    tx = min(max(x // ARENA_UNITS_PER_TILE, 0), GRID_W - 1)
    ty = min(max(y // ARENA_UNITS_PER_TILE, 0), GRID_H - 1)
    return tx, ty


def tile_for(event: Event, perspective: str) -> tuple[int, int] | None:
    """Tile of `event` in the frame of the `perspective` player."""
    if event.tile_x is None or event.tile_y is None:
        return None
    if perspective == "team":
        return event.tile_x, event.tile_y
    return GRID_W - 1 - event.tile_x, GRID_H - 1 - event.tile_y


def to_decision_steps(battle: Battle, side: str, step_ticks: int = TICKS_PER_SECOND):
    """Tick-level action stream for one player: one action per decision step.

    Returns a list of (tick, kind, card, tile) where kind is "wait" or "play".
    A step is "play" if the player plays a card in [tick, tick + step_ticks);
    if several plays fall in one step only the first is kept.
    """
    plays = []
    for e in battle.events:
        if e.side == side and e.kind == "play":
            plays.append(e)
    if not battle.events:
        return []
    end = battle.events[-1].tick + 1
    steps = []
    k = 0
    for tick in range(0, end, step_ticks):
        if k < len(plays) and plays[k].tick < tick + step_ticks:
            e = plays[k]
            steps.append((tick, "play", e.card, tile_for(e, side)))
            k += 1
            while k < len(plays) and plays[k].tick < tick + step_ticks:
                k += 1
        else:
            steps.append((tick, "wait", None, None))
    return steps
