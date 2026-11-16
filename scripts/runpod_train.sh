#!/usr/bin/env bash
# One-shot RunPod job: clone, install, download IL_Replay, preprocess, train, evaluate.
#
#   curl -fsSL https://raw.githubusercontent.com/fisherboy927/clash_ai/main/scripts/runpod_train.sh | bash
#
# Everything lives on the persistent volume under $WORK (default /workspace):
#   $WORK/clash_ai        repo clone (data/ and runs/ are symlinks to the two dirs below)
#   $WORK/clash_data      raw IL_Replay download + parsed + sequence files
#   $WORK/clash_runs      checkpoints and training logs
#   $WORK/clash_results   metrics.json and summary.md
# Steps whose outputs already exist are skipped, so rerunning after an interruption resumes
# at the first unfinished step (training itself restarts from scratch).
#
# Env overrides: WORK, REPO_URL, BRANCH, GITHUB_TOKEN (for a private repo), CONFIG,
# DEVICE (default cuda), LOADER_WORKERS, FORCE_TRAIN=1 (retrain even if a finished run exists).
# The data's legal status is unclear: never push data/ or runs/ anywhere.
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/fisherboy927/clash_ai.git}"
BRANCH="${BRANCH:-main}"
CONFIG="${CONFIG:-configs/warmup_gpu.json}"
DEVICE="${DEVICE:-cuda}"
REPO="$WORK/clash_ai"
DATA="$WORK/clash_data"
RUNS="$WORK/clash_runs"
RESULTS="$WORK/clash_results"
export HF_HOME="$WORK/hf_cache"
mkdir -p "$DATA" "$RUNS" "$RESULTS" "$HF_HOME"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# 1. Clone or update the repo.
if [ -n "${GITHUB_TOKEN:-}" ]; then
  CLONE_URL="${REPO_URL/https:\/\//https://x-access-token:${GITHUB_TOKEN}@}"
else
  CLONE_URL="$REPO_URL"
fi
if [ -d "$REPO/.git" ]; then
  log "updating repo ($BRANCH)"
  git -C "$REPO" fetch origin "$BRANCH"
  git -C "$REPO" checkout "$BRANCH"
  git -C "$REPO" reset --hard "origin/$BRANCH"
else
  log "cloning $REPO_URL ($BRANCH)"
  git clone --branch "$BRANCH" "$CLONE_URL" "$REPO"
fi
cd "$REPO"
for pair in "data:$DATA" "runs:$RUNS"; do
  name="${pair%%:*}"; target="${pair#*:}"
  if [ ! -L "$name" ]; then rm -rf "$name"; ln -s "$target" "$name"; fi
done

# 2. Python env: venv on the volume that reuses the template's CUDA torch when it works.
if [ ! -x "$WORK/venv/bin/python" ]; then
  log "creating venv"
  python3 -m venv --system-site-packages "$WORK/venv"
fi
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
if [ "$DEVICE" = "cuda" ] && ! python - <<'EOF'
import sys, torch
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
sys.exit(0 if torch.cuda.is_available() and (major, minor) >= (2, 4) else 1)
EOF
then
  log "installing pinned torch (template torch missing, too old, or without CUDA)"
  pip install -q -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -m pytest -q tests

# 3. Download (resumable, checksum-verified), parse to the shared format, build sequences.
log "downloading IL_Replay"
python -m clashai.download --root data/il
if [ ! -f data/il_parsed/parse_report.json ]; then
  log "parsing shards"
  python -m clashai.il_replay --root data/il --out data/il_parsed
fi
if [ ! -f data/il_seq/split_summary.json ]; then
  log "building sequences"
  python -m clashai.dataset --parsed data/il_parsed --out data/il_seq
fi
cat data/il_seq/split_summary.json

# 4. Train.
RUN_DIR="$(python -c "import json; print(json.load(open('$CONFIG'))['run_dir'])")"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/TRAINING_DONE" ] || [ "${FORCE_TRAIN:-0}" = "1" ]; then
  WORKERS="${LOADER_WORKERS:-$(( $(nproc) > 8 ? 6 : $(nproc) - 2 ))}"
  log "training ($CONFIG, $WORKERS loader workers)"
  python -m clashai.train --config "$CONFIG" --device "$DEVICE" --loader-workers "$WORKERS" 2>&1 | tee "$RUN_DIR/train.log"
  touch "$RUN_DIR/TRAINING_DONE"
fi

# 5. Evaluate against baselines and write results.
RESULT_NAME="$(basename "$RUN_DIR")"
log "evaluating"
python -m clashai.evaluate --run "$RUN_DIR" --data data/il_seq --costs data/il_parsed/card_costs.json \
  --out "results/$RESULT_NAME" --device "$DEVICE" 2>&1 | tee "$RUN_DIR/eval.log"
python -m clashai.summarize --metrics "results/$RESULT_NAME/metrics.json"
mkdir -p "$RESULTS/$RESULT_NAME"
cp "results/$RESULT_NAME/metrics.json" "results/$RESULT_NAME/summary.md" "$RUN_DIR/train_log.json" \
  data/il_parsed/parse_report.json data/il_seq/split_summary.json "$RESULTS/$RESULT_NAME/"
log "done: results in $RESULTS/$RESULT_NAME, checkpoint in $RUN_DIR/best.pt"
