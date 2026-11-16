# Clash_ai
An AI Clash Royale Player

## Warm-up model on replay data (private prototype)

A play-history-only transformer trained on the Hugging Face dataset
[VanguardX101/IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay)
(252k RoyaleAPI replays, 17.8M actions; no video, board inputs masked).

**Data rule:** the dataset is scraped and its legal status is unclear. Raw
data, parsed data and trained weights live only under `data/` and `runs/`
(gitignored; symlinks to the volume on RunPod). Never commit them, never
redistribute them, never publish a model trained on them. Only code, configs
and small result tables belong in this repo.

### What it does
- **Shared format** (`clashai/format.py`): 20 ticks/s, fixed 18 x 32 tile grid
  in each player's own frame; action = (wait/play/ability, card, tile, tick);
  observation = both players' play history plus optional per-event board-state
  vectors, masked when absent so video-derived board state can be added later.
- **Model** (`clashai/model.py`, ~15M params): causal transformer over events.
  At every event it predicts, for each player, the next own card (masked to the
  deck), its tile given the card, the time until that play (16 bins; P(play
  within 1 s) is the wait/play decision), and the win probability.
- **Splits** (`clashai/dataset.py`): IL_Replay has no dates or player IDs, so
  held-out sets are by battle: the last 3 parquet shards (`test_shard`, a rough
  proxy for a later collection batch), plus 4% random battles (`test_random`)
  and 2% for validation.
- **Baselines** (`clashai/evaluate.py`): random legal card (deck minus last 4
  own plays), most frequent legal card (optionally given the opponent's last
  card), most frequent tile per card (and per opponent lane), median delay per
  game phase and last actor, logistic win model on elixir and play-count
  difference. Metrics: card top-1/top-3, tile within-1/mean distance, timing
  MAE, wait/play AUC, win-prob AUC/Brier/ECE.

### Run on RunPod
1. **GPU:** one RTX 4090 (24 GB) is the best value; an L4 or A5000 also works,
   about 2x slower. Anything Ampere or newer gets bf16 autocast.
2. **Template:** the official RunPod PyTorch template, any recent CUDA 12.x
   version. The script keeps the template's CUDA torch if it is >= 2.4,
   otherwise installs the pinned one from `requirements-torch.txt`.
3. **Disk:** 20 GB volume mounted at `/workspace` (uses about 3 GB: 1.8 GB
   download, 0.7 GB preprocessed, 60 MB checkpoint); default 20 GB container disk.
4. **Command** (in the pod's web terminal):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/fisherboy927/clash_ai/main/scripts/runpod_train.sh | bash
   ```
   Before the PR is merged, use the branch instead:
   ```bash
   B=cursor/warmup-replay-model-b8a3
   curl -fsSL https://raw.githubusercontent.com/fisherboy927/clash_ai/$B/scripts/runpod_train.sh | BRANCH=$B bash
   ```
   Run it inside `tmux` so a closed browser tab doesn't stop it. Rerunning
   skips finished steps.
5. **Outputs:** `/workspace/clash_results/warmup_gpu/summary.md` (tables) and
   `metrics.json`; checkpoint at `/workspace/clash_runs/warmup_gpu/best.pt`.
   Keep the checkpoint on the volume; don't upload it anywhere public.

Expected time on a 4090: about 10 min download + preprocessing, 30-50 min
training (15 epochs, 32.5M tokens each, capped at 2 h by `max_hours`), 5 min
evaluation. The training log prints tokens/s and an ETA every 200 steps; if it
is much slower than expected, lower `epochs` in `configs/warmup_gpu.json`.

### Run locally (any device)
```bash
pip install -r requirements.txt   # plus torch for your platform
python -m clashai.download                         # -> data/il (checksum-verified)
python -m clashai.il_replay                        # -> data/il_parsed (shared format, shard by shard)
python -m clashai.dataset                          # -> data/il_seq
python -m clashai.train --config configs/warmup_gpu.json        # cuda / mps / cpu auto-detected
python -m clashai.evaluate && python -m clashai.summarize
```
`configs/pipeline_check.json` trains 3 steps for a quick end-to-end check;
`configs/smoke_cpu.json` is a small CPU run on a subset built with
`python -m clashai.dataset --out data/il_seq_smoke --max-train-battles 300 --max-eval-battles 100`.
Tests: `python -m pytest tests`.
