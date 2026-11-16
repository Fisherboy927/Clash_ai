from clashai.format import tile_for
from clashai.parse_royaleapi import normalize_orientation, parse_html


def marker(color, side, x, y, card, t):
    return (f'<div class="{color} marker" data-x="{x}" data-y="{y}" data-c="{card}" '
            f'data-t="{t}" data-i="1" data-s="{side}"><span>1</span></div>')


def page(team_y, opp_y):
    deck = ""
    for c in ("knight", "archers"):
        deck += f'<img class="replay_card" data-card="{c}" data-s="blue" src="x.png"/>'
        deck += f'<img class="replay_card" data-card="{c}" data-s="red" src="x.png"/>'
    marks = ""
    for i in range(6):
        marks += marker("blue", "t", 3500, team_y, "knight", 100 + 40 * i)
        marks += marker("red", "o", 14500, opp_y, "archers", 120 + 40 * i)
    return (f'<div class="battle_replay" data-tag="ABC123"><div class="markers">{marks}</div>'
            f'<div class="replay_stats">{deck}</div></div>')


URL = "https://royaleapi.com/data/replay?tag=ABC123&team_tags=T1&opponent_tags=O1&team_crowns=1&opponent_crowns=0"


def test_team_at_bottom_kept():
    bid, decks, crowns, tags, events = parse_html(page(5500, 26500), URL)
    assert bid == "ABC123"
    assert decks["team"] == ["knight", "archers"]
    assert crowns == {"team": 1, "opponent": 0}
    assert normalize_orientation(events) == "as_is"
    team_play = [e for e in events if e.side == "team"][0]
    assert (team_play.tile_x, team_play.tile_y) == (3, 5)
    opp_play = [e for e in events if e.side == "opponent"][0]
    assert tile_for(opp_play, "opponent") == (17 - 14, 31 - 26)


def test_team_at_top_rotated():
    _, _, _, _, events = parse_html(page(26500, 5500), URL)
    assert normalize_orientation(events) == "rotated"
    team_play = [e for e in events if e.side == "team"][0]
    assert (team_play.tile_x, team_play.tile_y) == (17 - 3, 31 - 26)
