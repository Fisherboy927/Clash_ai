"""Evaluate the warm-up model against simple baselines on held-out battles.

Usage: python -m clashai.evaluate --run runs/warmup --out results/warmup

Scoring points:
  card, tile : the query right before each own card play ("pre_play")
  timing     : every query that has a later own play
  win prob   : every query of a battle with a winner
Baselines are fit on the training split only.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .dataset import KIND_PLAY, NO_TILE, SIDE_OPP, SIDE_OWN, UNK
from .format import GRID_W
from .model import DELAY_EDGES, PLAY_SOON_BINS, PlayHistoryTransformer, collate, delay_bin, length_batches
from .train import load_split

RECENT_TICKS = 100
PHASES = ["0-1 min", "1-2 min", "2-3 min", "overtime"]


def phase_of(tick):
    if tick < 1200:
        return 0
    if tick < 2400:
        return 1
    if tick < 3600:
        return 2
    return 3


def context_features(arrays, costs, inv_vocab, with_legal=True):
    """Per-token situation features used by baselines and metric breakdowns."""
    n_tok = len(arrays["card"])
    legal = None
    if with_legal:
        legal = [None] * n_tok
    opp_last = np.zeros(n_tok, dtype=np.int64)
    opp_lane = np.zeros(n_tok, dtype=np.int64)  # 0 none recent, 1 left, 2 right
    last_side = np.zeros(n_tok, dtype=np.int64)
    phase = np.zeros(n_tok, dtype=np.int64)
    elixir_diff = np.zeros(n_tok, dtype=np.float32)
    play_diff = np.zeros(n_tok, dtype=np.float32)
    for s in range(len(arrays["length"])):
        o = int(arrays["offset"][s])
        n = int(arrays["length"][s])
        deck = []
        for c in arrays["deck"][s]:
            if c > UNK:
                deck.append(int(c))
        recent_own = []
        last_opp_card = 0
        last_opp_tile = NO_TILE
        last_opp_tick = -10_000
        spent = 0.0
        plays = 0
        for i in range(n):
            g = o + i
            card = int(arrays["card"][g])
            side = int(arrays["side"][g])
            kind = int(arrays["kind"][g])
            tick = int(arrays["tick"][g])
            if kind == KIND_PLAY:
                cost = costs.get(inv_vocab.get(card, ""), 3)
                if side == SIDE_OWN:
                    if card in recent_own:
                        recent_own.remove(card)
                    recent_own.append(card)
                    if len(recent_own) > 4:
                        recent_own.pop(0)
                    spent += cost
                    plays += 1
                else:
                    last_opp_card = card
                    last_opp_tile = int(arrays["tile"][g])
                    last_opp_tick = tick
                    spent -= cost
                    plays -= 1
            if with_legal:
                leg = []
                for c in deck:
                    if c not in recent_own:
                        leg.append(c)
                legal[g] = leg
            opp_last[g] = last_opp_card
            if tick - last_opp_tick <= RECENT_TICKS and last_opp_tile < NO_TILE:
                if (last_opp_tile % GRID_W) < GRID_W // 2:
                    opp_lane[g] = 1
                else:
                    opp_lane[g] = 2
            last_side[g] = side
            phase[g] = phase_of(tick)
            elixir_diff[g] = spent
            play_diff[g] = plays
    return {"legal": legal, "opp_last": opp_last, "opp_lane": opp_lane, "last_side": last_side,
            "phase": phase, "elixir_diff": elixir_diff, "play_diff": play_diff}


def tile_xy(t):
    return t % GRID_W, t // GRID_W


def tile_metrics(pred, true):
    px, py = tile_xy(pred)
    tx, ty = tile_xy(true)
    dist = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)
    cheb = np.maximum(np.abs(px - tx), np.abs(py - ty))
    return {"exact": float((pred == true).mean()), "within_1_tile": float((cheb <= 1).mean()),
            "mean_dist_tiles": float(dist.mean()), "median_dist_tiles": float(np.median(dist))}


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    table = []
    for b in range(bins):
        lo = edges[b]
        hi = edges[b + 1]
        if b == bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        gap = abs(p[m].mean() - y[m].mean())
        total += m.mean() * gap
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                      "win_rate": float(y[m].mean())})
    return float(total), table


def win_metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    e, table = ece(p, y)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"auc": auc, "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "ece": e}, table


class Baselines:
    def __init__(self, train, ctx):
        pre = np.where(train["pre_play"] == 1)[0]
        self.card_freq = collections.Counter()
        self.card_given_opp = collections.defaultdict(collections.Counter)
        self.tile_by_card = collections.defaultdict(collections.Counter)
        self.tile_by_card_lane = collections.defaultdict(collections.Counter)
        for g in pre:
            c = int(train["t_card"][g])
            t = int(train["t_tile"][g])
            self.card_freq[c] += 1
            self.card_given_opp[int(ctx["opp_last"][g])][c] += 1
            self.tile_by_card[c][t] += 1
            self.tile_by_card_lane[(c, int(ctx["opp_lane"][g]))][t] += 1
        total = sum(self.card_freq.values())
        self.card_p = {}
        for c in self.card_freq:
            self.card_p[c] = self.card_freq[c] / total
        self.tile_mode = {}
        for c in self.tile_by_card:
            self.tile_mode[c] = self.tile_by_card[c].most_common(1)[0][0]
        self.tile_mode_lane = {}
        for k in self.tile_by_card_lane:
            if sum(self.tile_by_card_lane[k].values()) >= 20:
                self.tile_mode_lane[k] = self.tile_by_card_lane[k].most_common(1)[0][0]

        has = np.where(train["t_delay"] >= 0)[0]
        self.delay_median = float(np.median(train["t_delay"][has]))
        groups = collections.defaultdict(list)
        for g in has:
            groups[(int(ctx["phase"][g]), int(ctx["last_side"][g]))].append(int(train["t_delay"][g]))
        self.delay_cond = {}
        self.soon_cond = {}
        for k in groups:
            v = np.array(groups[k])
            self.delay_cond[k] = float(np.median(v))
            self.soon_cond[k] = float((v < DELAY_EDGES[PLAY_SOON_BINS]).mean())

        win_rows = np.where(train["win_tok"] >= 0)[0]
        rng = np.random.default_rng(0)
        sub = rng.choice(win_rows, size=min(300_000, len(win_rows)), replace=False)
        self.win_lr = LogisticRegression(max_iter=1000)
        self.win_lr.fit(self.win_features(train, ctx, sub), train["win_tok"][sub])

    @staticmethod
    def win_features(arrays, ctx, idx):
        t = arrays["tick"][idx] / 1200.0
        e = ctx["elixir_diff"][idx]
        p = ctx["play_diff"][idx]
        return np.stack([e, p, t, e * t], 1)

    def card_scores(self, legal, opp_last, mode):
        best = None
        best_s = -1.0
        ranked = []
        for c in legal:
            if mode == "freq":
                s = self.card_p.get(c, 0.0)
            else:
                cond = self.card_given_opp.get(opp_last, {})
                n = sum(cond.values()) if cond else 0
                s = (cond.get(c, 0) + 20 * self.card_p.get(c, 0.0)) / (n + 20)
            ranked.append((s, c))
            if s > best_s:
                best_s = s
                best = c
        ranked.sort(reverse=True)
        return best, ranked


def subset_sequences(arrays, keep):
    """Arrays restricted to the sequences where keep is True."""
    seqs = np.where(keep)[0]
    out = {}
    token_fields = ["card", "side", "kind", "tile", "tick", "t_card", "t_tile", "t_delay", "pre_play"]
    idx = []
    for s in seqs:
        o = int(arrays["offset"][s])
        idx.append(np.arange(o, o + int(arrays["length"][s])))
    idx = np.concatenate(idx)
    for k in token_fields:
        out[k] = arrays[k][idx]
    for k in ("length", "deck", "win", "battle_index", "player_seen_in_train"):
        out[k] = arrays[k][seqs]
    out["offset"] = np.concatenate([[0], np.cumsum(out["length"])[:-1]]).astype(np.int64)
    return out


def hand_mask(ctx, arrays, ids, T, vocab_size):
    """Boolean (B, T, V) mask of cards in the inferred hand (deck minus last 4 own plays)."""
    m = torch.zeros(len(ids), T, vocab_size, dtype=torch.bool)
    for b in range(len(ids)):
        s = ids[b]
        o = int(arrays["offset"][s])
        n = int(arrays["length"][s])
        for i in range(n):
            leg = ctx["legal"][o + i]
            if leg:
                m[b, i, leg] = True
            else:
                m[b, i, :] = True
    return m


def model_predictions(model, arrays, ctx):
    n_tok = len(arrays["card"])
    pred = {
        "card_top1_hand": np.zeros(n_tok, dtype=np.int64),
        "card_top3_hit_hand": np.zeros(n_tok, dtype=bool),
        "card_top1": np.zeros(n_tok, dtype=np.int64),
        "card_top3_hit": np.zeros(n_tok, dtype=bool),
        "card_nll": np.zeros(n_tok, dtype=np.float32),
        "tile_true_card": np.zeros(n_tok, dtype=np.int64),
        "tile_pred_card": np.zeros(n_tok, dtype=np.int64),
        "delay_median": np.zeros(n_tok, dtype=np.float32),
        "p_soon": np.zeros(n_tok, dtype=np.float32),
        "p_win": np.zeros(n_tok, dtype=np.float32),
    }
    has = np.where(arrays["t_delay"] >= 0)[0]
    bins = delay_bin(torch.from_numpy(arrays["t_delay"][has].astype(np.int64))).numpy()
    rep = np.zeros(len(DELAY_EDGES), dtype=np.float32)
    for b in range(len(DELAY_EDGES)):
        m = bins == b
        if m.any():
            rep[b] = np.median(arrays["t_delay"][has][m])
        else:
            rep[b] = DELAY_EDGES[b]
    model.eval()
    with torch.no_grad():
        for ids in length_batches(arrays["length"], 16384):
            batch = collate(arrays, ids)
            out = model(batch)
            logp = torch.log_softmax(out["card"], -1)
            top3 = logp.topk(3, -1).indices
            top1 = top3[..., 0]
            tgt = batch["t_card"].clamp(min=0)
            nll = -logp.gather(-1, tgt[..., None]).squeeze(-1)
            hit3 = (top3 == tgt[..., None]).any(-1)
            hm = hand_mask(ctx, arrays, ids, logp.shape[1], logp.shape[2])
            top3_h = logp.masked_fill(~hm, -1e9).topk(3, -1).indices
            hit3_h = (top3_h == tgt[..., None]).any(-1)
            tile_t = model.tile_logits(out["h"], batch["t_card"]).argmax(-1)
            tile_p = model.tile_logits(out["h"], top1).argmax(-1)
            dprob = torch.softmax(out["delay"], -1)
            cdf = dprob.cumsum(-1)
            med_bin = (cdf < 0.5).sum(-1).clamp(max=len(DELAY_EDGES) - 1)
            p_soon = dprob[..., :PLAY_SOON_BINS].sum(-1)
            p_win = torch.sigmoid(out["win"])
            for b in range(len(ids)):
                s = ids[b]
                o = int(arrays["offset"][s])
                n = int(arrays["length"][s])
                sl = slice(o, o + n)
                pred["card_top1"][sl] = top1[b, :n].numpy()
                pred["card_top1_hand"][sl] = top3_h[b, :n, 0].numpy()
                pred["card_top3_hit_hand"][sl] = hit3_h[b, :n].numpy()
                pred["card_top3_hit"][sl] = hit3[b, :n].numpy()
                pred["card_nll"][sl] = nll[b, :n].numpy()
                pred["tile_true_card"][sl] = tile_t[b, :n].numpy()
                pred["tile_pred_card"][sl] = tile_p[b, :n].numpy()
                pred["delay_median"][sl] = rep[med_bin[b, :n].numpy()]
                pred["p_soon"][sl] = p_soon[b, :n].numpy()
                pred["p_win"][sl] = p_win[b, :n].numpy()
    return pred


def add_token_win(arrays):
    win_tok = np.zeros(len(arrays["card"]), dtype=np.int64)
    for s in range(len(arrays["length"])):
        o = int(arrays["offset"][s])
        n = int(arrays["length"][s])
        win_tok[o:o + n] = arrays["win"][s]
    arrays["win_tok"] = win_tok


def evaluate_split(name, arrays, ctx, base, pred):
    res = {"split": name, "battles": int(len(np.unique(arrays["battle_index"]))),
           "player_sequences": int(len(arrays["length"]))}
    pre = np.where((arrays["pre_play"] == 1) & (arrays["t_card"] > UNK))[0]
    res["card_scoring_points"] = int(len(pre))
    tgt = arrays["t_card"][pre].astype(np.int64)

    in_cycle = np.zeros(len(pre), dtype=bool)
    rand_top1 = np.zeros(len(pre))
    rand_top3 = np.zeros(len(pre))
    freq_top1 = np.zeros(len(pre), dtype=bool)
    freq_top3 = np.zeros(len(pre), dtype=bool)
    cond_top1 = np.zeros(len(pre), dtype=bool)
    cond_top3 = np.zeros(len(pre), dtype=bool)
    cond_pred = np.zeros(len(pre), dtype=np.int64)
    for k in range(len(pre)):
        g = pre[k]
        leg = ctx["legal"][g]
        t = int(tgt[k])
        if t in leg:
            in_cycle[k] = True
            rand_top1[k] = 1.0 / len(leg)
            rand_top3[k] = min(3, len(leg)) / len(leg)
        best, ranked = base.card_scores(leg, int(ctx["opp_last"][g]), "freq")
        freq_top1[k] = best == t
        top = []
        for j in range(min(3, len(ranked))):
            top.append(ranked[j][1])
        freq_top3[k] = t in top
        best, ranked = base.card_scores(leg, int(ctx["opp_last"][g]), "cond")
        cond_pred[k] = best if best is not None else 0
        cond_top1[k] = best == t
        top = []
        for j in range(min(3, len(ranked))):
            top.append(ranked[j][1])
        cond_top3[k] = t in top
    res["target_in_cycle_hand_rate"] = float(in_cycle.mean())
    res["card"] = {
        "random_legal (deck minus last 4 own plays)": {"top1": float(rand_top1.mean()), "top3": float(rand_top3.mean())},
        "most_frequent_legal": {"top1": float(freq_top1.mean()), "top3": float(freq_top3.mean())},
        "most_frequent_legal_given_opp_last_card": {"top1": float(cond_top1.mean()), "top3": float(cond_top3.mean())},
        "model": {"top1": float((pred["card_top1"][pre] == tgt).mean()), "top3": float(pred["card_top3_hit"][pre].mean()),
                  "nll": float(pred["card_nll"][pre].mean())},
        "model + hand mask": {"top1": float((pred["card_top1_hand"][pre] == tgt).mean()),
                              "top3": float(pred["card_top3_hit_hand"][pre].mean())},
    }
    seq_of_tok = np.repeat(np.arange(len(arrays["length"])), arrays["length"])
    seen = arrays["player_seen_in_train"][seq_of_tok[pre]] == 1
    by_seen = {}
    for label, m in (("player_seen_in_train", seen), ("player_not_in_train", ~seen)):
        if m.sum() < 100:
            continue
        by_seen[label] = {"n": int(m.sum()),
                          "model_top1": float((pred["card_top1"][pre][m] == tgt[m]).mean()),
                          "model_hand_top1": float((pred["card_top1_hand"][pre][m] == tgt[m]).mean()),
                          "cond_freq_top1": float(cond_top1[m].mean())}
    res["card_top1_by_player_seen"] = by_seen

    true_tile = arrays["t_tile"][pre].astype(np.int64)
    rand_dist = np.zeros(len(pre))
    rand_w1 = np.zeros(len(pre))
    rand_exact = np.zeros(len(pre))
    mode_t = np.zeros(len(pre), dtype=np.int64)
    mode_lane_t = np.zeros(len(pre), dtype=np.int64)
    support = {}
    for c in base.tile_by_card:
        ts = np.array(list(base.tile_by_card[c].keys()))
        support[c] = (ts % GRID_W, ts // GRID_W)
    all_tiles = np.arange(NO_TILE)
    all_support = (all_tiles % GRID_W, all_tiles // GRID_W)
    for k in range(len(pre)):
        c = int(tgt[k])
        sx, sy = support.get(c, all_support)
        tx, ty = tile_xy(true_tile[k])
        rand_dist[k] = np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2).mean()
        rand_w1[k] = (np.maximum(np.abs(sx - tx), np.abs(sy - ty)) <= 1).mean()
        rand_exact[k] = ((sx == tx) & (sy == ty)).mean()
        mode_t[k] = base.tile_mode.get(c, 8 * GRID_W + 9)
        mode_lane_t[k] = base.tile_mode_lane.get((c, int(ctx["opp_lane"][pre[k]])), mode_t[k])
    cond_card_tile = np.zeros(len(pre), dtype=np.int64)
    for k in range(len(pre)):
        c = int(cond_pred[k])
        cond_card_tile[k] = base.tile_mode_lane.get((c, int(ctx["opp_lane"][pre[k]])), base.tile_mode.get(c, 8 * GRID_W + 9))
    res["tile_given_true_card"] = {
        "random_tile_where_card_was_ever_placed": {"exact": float(rand_exact.mean()), "within_1_tile": float(rand_w1.mean()),
                                                   "mean_dist_tiles": float(rand_dist.mean())},
        "most_frequent_tile_for_card": tile_metrics(mode_t, true_tile),
        "most_frequent_tile_for_card_and_opp_lane": tile_metrics(mode_lane_t, true_tile),
        "model": tile_metrics(pred["tile_true_card"][pre], true_tile),
    }
    res["tile_end_to_end (predicted card)"] = {
        "cond_freq_card + most_frequent_tile": tile_metrics(cond_card_tile, true_tile),
        "model": tile_metrics(pred["tile_pred_card"][pre], true_tile),
    }

    has = np.where(arrays["t_delay"] >= 0)[0]
    res["timing_scoring_points"] = int(len(has))
    d_true = arrays["t_delay"][has] / 20.0
    const = np.full(len(has), base.delay_median / 20.0)
    cond = np.zeros(len(has))
    soon_b = np.zeros(len(has))
    for k in range(len(has)):
        key = (int(ctx["phase"][has[k]]), int(ctx["last_side"][has[k]]))
        cond[k] = base.delay_cond.get(key, base.delay_median) / 20.0
        soon_b[k] = base.soon_cond.get(key, 0.5)
    d_model = pred["delay_median"][has] / 20.0
    soon_y = (arrays["t_delay"][has] < DELAY_EDGES[PLAY_SOON_BINS]).astype(int)

    def tm(p):
        err = np.abs(p - d_true)
        return {"mae_s": float(err.mean()), "median_abs_err_s": float(np.median(err)),
                "within_0.5s": float((err <= 0.5).mean()), "within_1s": float((err <= 1.0).mean())}

    res["timing_delay_to_next_own_play"] = {
        "constant_median": tm(const),
        "median_by_phase_and_last_actor": tm(cond),
        "model": tm(d_model),
    }
    res["wait_or_play_within_1s"] = {
        "base_rate": float(soon_y.mean()),
        "rate_by_phase_and_last_actor_auc": float(roc_auc_score(soon_y, soon_b)),
        "model_auc": float(roc_auc_score(soon_y, pred["p_soon"][has])),
    }

    wrows = np.where(arrays["win_tok"] >= 0)[0]
    y = arrays["win_tok"][wrows]
    p_lr = base.win_lr.predict_proba(Baselines.win_features(arrays, ctx, wrows))[:, 1]
    m_const, _ = win_metrics(np.full(len(wrows), 0.5), y)
    m_lr, _ = win_metrics(p_lr, y)
    m_model, table = win_metrics(pred["p_win"][wrows], y)
    res["win_prob"] = {"constant_0.5": m_const, "logistic_on_elixir_and_play_count_diff": m_lr, "model": m_model}
    res["win_calibration_table_model"] = table
    by_phase = {}
    for ph in range(4):
        m = ctx["phase"][wrows] == ph
        if m.sum() < 100:
            continue
        a, _ = win_metrics(pred["p_win"][wrows][m], y[m])
        b, _ = win_metrics(p_lr[m], y[m])
        by_phase[PHASES[ph]] = {"n": int(m.sum()), "model_auc": a["auc"], "model_ece": a["ece"],
                                "logistic_auc": b["auc"]}
    res["win_prob_by_phase"] = by_phase

    card_by_phase = {}
    for ph in range(4):
        m = ctx["phase"][pre] == ph
        if m.sum() < 100:
            continue
        card_by_phase[PHASES[ph]] = {"n": int(m.sum()),
                                     "model_top1": float((pred["card_top1"][pre][m] == tgt[m]).mean()),
                                     "cond_freq_top1": float(cond_top1[m].mean())}
    res["card_top1_by_phase"] = card_by_phase
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/warmup")
    ap.add_argument("--data", default="data/seq")
    ap.add_argument("--costs", default="data/hf/card_costs.json")
    ap.add_argument("--out", default="results/warmup")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    with open(os.path.join(args.data, "vocab.json")) as f:
        vocab = json.load(f)
    inv_vocab = {}
    for k in vocab:
        inv_vocab[vocab[k]] = k
    with open(args.costs) as f:
        costs = json.load(f)

    ckpt = torch.load(os.path.join(args.run, "best.pt"), weights_only=False)
    model = PlayHistoryTransformer(ckpt["vocab_size"], **ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])

    train = load_split(args.data, "train")
    add_token_win(train)
    print("fitting baselines", flush=True)
    base = Baselines(train, context_features(train, costs, inv_vocab, with_legal=False))
    del train

    all_res = {"checkpoint_epoch": ckpt["epoch"]}
    for name in ("test_time", "test_player", "test_random"):
        arrays = load_split(args.data, name)
        if name == "test_player":
            arrays = subset_sequences(arrays, arrays["player_seen_in_train"] == 0)
        add_token_win(arrays)
        ctx = context_features(arrays, costs, inv_vocab)
        print("predicting", name, flush=True)
        pred = model_predictions(model, arrays, ctx)
        all_res[name] = evaluate_split(name, arrays, ctx, base, pred)
        print(json.dumps(all_res[name], indent=1), flush=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
