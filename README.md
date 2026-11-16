# Clash_ai
An AI Clash Royale Player

## Warm-up model on replay data (private prototype)

A first model trained only on the Hugging Face dataset
`Cochon123/clash-royale-replays` (RoyaleAPI replay captures, no video).

**Data rule:** the dataset's legal status is unclear. Raw data, parsed data and
trained weights live only under `data/` and `runs/`, which are gitignored.
Never commit them, never redistribute them, and never publish a model trained
on them. Only code, configs and small result tables belong in this repo.

### Shared format (`clashai/format.py`)
- Time: integer ticks, 20 ticks/s. Board: fixed 18 x 32 tile grid, stored in
  each player's own frame (row 0 = own back row).
- Action: (wait/play/ability, card, tile, tick).
- Observation: both players' play history; optional board-state vectors per
  event, masked when absent (always, for replays), so video-derived board state
  can be added later on the same network.

### Pipeline
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn

# 1. Download (latest snapshot + cleaned-legacy) into the gitignored data/ dir
B=https://huggingface.co/datasets/Cochon123/clash-royale-replays/resolve/main
mkdir -p data/hf/snapshots data/hf/cleaned-legacy data/hf/snap
curl -L -o data/hf/snapshots/data-2026-11-01.tar.gz $B/snapshots/data-2026-11-01.tar.gz
for f in events.jsonl matches.jsonl manifest.json; do curl -L -o data/hf/cleaned-legacy/$f $B/cleaned-legacy/$f; done
curl -L -o data/hf/card_costs.json $B/card_costs.json
tar -xzf data/hf/snapshots/data-2026-11-01.tar.gz -C data/hf/snap

# 2. Parse replays into the shared format, then build token sequences + splits
python -m clashai.parse_royaleapi      # -> data/parsed/ (+ parse_report.json)
python -m clashai.dataset              # -> data/seq/

# 3. Train (CPU, ~35 min on 4 cores), evaluate against baselines
python -m clashai.train --config configs/warmup.json   # -> runs/warmup/
python -m clashai.evaluate --run runs/warmup --out results/warmup
python -m clashai.summarize --metrics results/warmup/metrics.json

python -m pytest tests
```

Results: `results/warmup/summary.md`.
