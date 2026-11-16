"""Small causal transformer over play history.

Inputs per token: card, side (own/opponent), kind (start/play/ability), tile
(own frame), time-in-battle bucket, time-since-previous-token bucket, plus an
optional board-state vector that is masked when unavailable (always, for
replay data). The player's own deck is added to every token as a conditioning
vector, since a player always knows their deck.

Heads (at every token = query point):
  card  : next own card, logits restricted to the own deck
  tile  : tile of that play given the card (576 classes, own frame)
  delay : ticks until the next own play, 16 bins; P(play within 1 s) is the
          sum of the first three bins, which is the wait/play decision
  win   : probability that this player wins
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import NO_TILE, PAD, UNK
from .format import GRID_W, GRID_H

DELAY_EDGES = [0, 5, 10, 20, 30, 40, 60, 80, 100, 120, 160, 200, 260, 340, 440, 600]
PLAY_SOON_BINS = 3
N_DELAY_BINS = len(DELAY_EDGES)
N_TICK_BUCKETS = 361
N_GAP_BUCKETS = 16


def delay_bin(delay: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(DELAY_EDGES[1:], device=delay.device)
    return torch.bucketize(delay, edges, right=True)


def gap_bucket(gap: torch.Tensor) -> torch.Tensor:
    return delay_bin(gap.clamp(min=0))


class PlayHistoryTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, n_layers=4, n_heads=4, d_ff=512,
                 dropout=0.1, board_dim=32, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.board_dim = board_dim
        self.card_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.side_emb = nn.Embedding(3, d_model)
        self.kind_emb = nn.Embedding(3, d_model)
        self.tile_emb = nn.Embedding(NO_TILE + 1, d_model)
        self.tick_emb = nn.Embedding(N_TICK_BUCKETS, d_model)
        self.gap_emb = nn.Embedding(N_GAP_BUCKETS, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.deck_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.board_proj = nn.Linear(board_dim, d_model)
        self.board_missing = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.card_head = nn.Linear(d_model, vocab_size)
        self.tile_card_emb = nn.Embedding(vocab_size, d_model)
        self.tile_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, GRID_W * GRID_H))
        self.delay_head = nn.Linear(d_model, N_DELAY_BINS)
        self.win_head = nn.Linear(d_model, 1)

    def forward(self, batch):
        card = batch["card"]
        B, T = card.shape
        tick = batch["tick"]
        gap = torch.zeros_like(tick)
        gap[:, 1:] = tick[:, 1:] - tick[:, :-1]
        x = (self.card_emb(card) + self.side_emb(batch["side"]) + self.kind_emb(batch["kind"])
             + self.tile_emb(batch["tile"]) + self.tick_emb((tick // 20).clamp(max=N_TICK_BUCKETS - 1))
             + self.gap_emb(gap_bucket(gap)) + self.pos_emb(torch.arange(T, device=card.device))[None])
        deck = batch["deck"]
        deck_vec = self.deck_emb(deck).sum(1) / (deck != PAD).sum(1, keepdim=True).clamp(min=1)
        x = x + deck_vec[:, None, :]
        board = batch.get("board")
        board_mask = batch.get("board_mask")
        if board is None:
            x = x + self.board_missing
        else:
            m = board_mask[..., None].float()
            x = x + m * self.board_proj(board) + (1 - m) * self.board_missing
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=card.device), diagonal=1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=batch["pad"])
        h = self.norm(h)

        card_logits = self.card_head(h)
        legal = torch.zeros(B, self.vocab_size, dtype=torch.bool, device=card.device)
        legal.scatter_(1, deck.long(), True)
        legal[:, PAD] = False
        legal[:, UNK] = False
        card_logits = card_logits.masked_fill(~legal[:, None, :], -1e9)
        return {
            "h": h,
            "card": card_logits,
            "delay": self.delay_head(h),
            "win": self.win_head(h).squeeze(-1),
        }

    def tile_logits(self, h, card):
        return self.tile_head(torch.cat([h, self.tile_card_emb(card.clamp(min=0))], -1))


def collate(arrays, seq_ids):
    """Pad a list of sequences (by index into the split arrays) into a batch."""
    lengths = arrays["length"][seq_ids]
    T = int(lengths.max())
    B = len(seq_ids)
    fields = {"card": PAD, "side": 0, "kind": 0, "tile": NO_TILE, "tick": 0,
              "t_card": -1, "t_tile": -1, "t_delay": -1, "pre_play": 0}
    out = {}
    for f in fields:
        out[f] = np.full((B, T), fields[f], dtype=np.int64)
    for b in range(B):
        s = seq_ids[b]
        o = arrays["offset"][s]
        n = arrays["length"][s]
        for f in fields:
            out[f][b, :n] = arrays[f][o:o + n]
    batch = {}
    for f in fields:
        batch[f] = torch.from_numpy(out[f])
    pad = np.ones((B, T), dtype=bool)
    for b in range(B):
        pad[b, :lengths[b]] = False
    batch["pad"] = torch.from_numpy(pad)
    batch["deck"] = torch.from_numpy(arrays["deck"][seq_ids].astype(np.int64))
    win = arrays["win"][seq_ids].astype(np.int64)
    batch["win"] = torch.from_numpy(np.repeat(win[:, None], T, axis=1))
    return batch


def length_batches(lengths, max_tokens, rng=None):
    """Group sequences of similar length into batches of about max_tokens."""
    order = np.argsort(lengths, kind="stable")
    if rng is not None:
        noise = rng.random(len(lengths)) * 8
        order = np.argsort(lengths + noise, kind="stable")
    batches = []
    cur = []
    cur_max = 0
    for i in order:
        L = int(lengths[i])
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            batches.append(np.array(cur))
            cur = []
            cur_max = 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(np.array(cur))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def losses(model, out, batch, weights):
    valid = ~batch["pad"]
    has_next = (batch["t_delay"] >= 0) & valid
    card_ok = has_next & (batch["t_card"] > UNK)
    l_card = F.cross_entropy(out["card"][card_ok], batch["t_card"][card_ok])
    tile_ok = card_ok & (batch["t_tile"] >= 0) & (batch["t_tile"] < NO_TILE)
    tl = model.tile_logits(out["h"][tile_ok], batch["t_card"][tile_ok])
    l_tile = F.cross_entropy(tl, batch["t_tile"][tile_ok])
    l_delay = F.cross_entropy(out["delay"][has_next], delay_bin(batch["t_delay"][has_next]))
    win_ok = valid & (batch["win"] >= 0)
    l_win = F.binary_cross_entropy_with_logits(out["win"][win_ok], batch["win"][win_ok].float())
    total = weights["card"] * l_card + weights["tile"] * l_tile + weights["delay"] * l_delay + weights["win"] * l_win
    parts = {"card": l_card.item(), "tile": l_tile.item(), "delay": l_delay.item(), "win": l_win.item()}
    return total, parts


def count_params(model):
    n = 0
    for p in model.parameters():
        n += p.numel()
    return n


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * 0.5 * (1 + math.cos(math.pi * progress))
