"""Render results/<run>/metrics.json as a small markdown table.

Usage: python -m clashai.summarize --metrics results/warmup_gpu/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os

SPLITS = ["test_shard", "test_random"]


def pct(x):
    return f"{100 * x:.1f}%"


def row(name, cells):
    return "| " + name + " | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/warmup_gpu/metrics.json")
    args = ap.parse_args()
    with open(args.metrics) as f:
        m = json.load(f)
    splits = []
    for s in SPLITS:
        if s in m:
            splits.append(s)
    header = "| | " + " | ".join(splits) + " |"
    sep = "|---|" + "---|" * len(splits)
    lines = []

    lines.append("Held-out sets: " + ", ".join(
        f"{s} = {m[s]['battles']} battles / {m[s]['card_scoring_points']} plays" for s in splits))
    lines.append("")
    lines.append("**Next card, top-1 / top-3 (query right before each play)**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for name in m[splits[0]]["card"]:
        cells = []
        for s in splits:
            c = m[s]["card"][name]
            cells.append(f"{pct(c['top1'])} / {pct(c['top3'])}")
        lines.append(row(name, cells))
    lines.append("")

    lines.append("**Tile given the true card: within 1 tile / mean distance (tiles)**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for name in m[splits[0]]["tile_given_true_card"]:
        cells = []
        for s in splits:
            c = m[s]["tile_given_true_card"][name]
            cells.append(f"{pct(c['within_1_tile'])} / {c['mean_dist_tiles']:.2f}")
        lines.append(row(name, cells))
    for name in m[splits[0]]["tile_end_to_end (predicted card)"]:
        cells = []
        for s in splits:
            c = m[s]["tile_end_to_end (predicted card)"][name]
            cells.append(f"{pct(c['within_1_tile'])} / {c['mean_dist_tiles']:.2f}")
        lines.append(row("end-to-end: " + name, cells))
    lines.append("")

    lines.append("**Timing: delay to next own play, MAE (s) / within 1 s; wait-vs-play (play within 1 s) AUC**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for name in m[splits[0]]["timing_delay_to_next_own_play"]:
        cells = []
        for s in splits:
            c = m[s]["timing_delay_to_next_own_play"][name]
            cells.append(f"{c['mae_s']:.2f} / {pct(c['within_1s'])}")
        lines.append(row(name, cells))
    cells = []
    for s in splits:
        w = m[s]["wait_or_play_within_1s"]
        cells.append(f"baseline {w['rate_by_phase_and_last_actor_auc']:.3f}, model {w['model_auc']:.3f}")
    lines.append(row("wait/play AUC", cells))
    lines.append("")

    lines.append("**Win probability (all query points): AUC / Brier / ECE**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for name in m[splits[0]]["win_prob"]:
        cells = []
        for s in splits:
            c = m[s]["win_prob"][name]
            cells.append(f"{c['auc']:.3f} / {c['brier']:.3f} / {c['ece']:.3f}")
        lines.append(row(name, cells))
    lines.append("")
    lines.append("**Win-probability AUC by game phase (test_shard)**")
    lines.append("")
    lines.append("| phase | n | model AUC | model ECE | logistic AUC |")
    lines.append("|---|---|---|---|---|")
    for ph, c in m["test_shard"]["win_prob_by_phase"].items():
        lines.append(f"| {ph} | {c['n']} | {c['model_auc']:.3f} | {c['model_ece']:.3f} | {c['logistic_auc']:.3f} |")
    lines.append("")

    out = os.path.join(os.path.dirname(args.metrics), "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
