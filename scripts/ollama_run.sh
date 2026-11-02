#!/usr/bin/env bash
# Run the harness against a LOCAL Ollama server on a small fixed split of the training data.
# usage: [SPLIT=train20|purchase12] scripts/ollama_run.sh [run-name] [extra `datamate run` args, e.g. --limit 1]
#   SPLIT=train20     20 questions, every answer type, one per shop (docs/train20_ids.txt)    [default]
#   SPLIT=purchase12  12 purchase questions, one per shop (docs/purchase12_ids.txt)
#   SPLIT=eval5       5 questions: the failed Omachi one + 4 purchase questions on 3 layouts (docs/eval5_ids.txt)
#   SPLIT=dev5        5 dev300 questions, one per answer type and shop (docs/dev5_ids.txt)
# P2: DATAMATE_P2=fast (default: one spec call, checks in code) | legacy; DATAMATE_PLAN_MODE=spec (default) | call | none
# A/B of the structure cards (data/structure.py) on the same questions:
#   SPLIT=purchase12 DATAMATE_STRUCTURE_CARDS=0 scripts/ollama_run.sh p12-cards-off
#   SPLIT=purchase12 DATAMATE_STRUCTURE_CARDS=1 scripts/ollama_run.sh p12-cards-on
# Env: WORKERS (default 1: one local GPU, parallel requests only queue), OLLAMA_MODEL, OLLAMA_BASE_URL,
#      DATAMATE_PLAN_THINKING (default 0 here: the plan call's thinking is ~12k tokens, minutes at laptop speed),
#      DATAMATE_EXPLAINER_CALLS / _CHECKS (default 3 / 1 here, 5 / 2 in the code: the Explainer was 48% of a question's time on
#      the first local run). Set them back to 5 / 2 to compare with BASELINE.md.
# Resume an interrupted run:  scripts/ollama_run.sh <name> --resume-run result/runs/<datetime>_<name>
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${SPLIT:-train20}
NAME=${1:-ollama-$SPLIT}; shift || true
export DATAMATE_BACKEND=ollama
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://localhost:11434/v1}
export OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3.5-datamate}
export DATAMATE_WORKERS=${WORKERS:-1}
export DATAMATE_PLAN_THINKING=${DATAMATE_PLAN_THINKING:-0}
export DATAMATE_STRUCTURE_CARDS=${DATAMATE_STRUCTURE_CARDS:-1}
export DATAMATE_PURCHASE_ENGINE=${DATAMATE_PURCHASE_ENGINE:-1}
export DATAMATE_EXPLAINER_CALLS=${DATAMATE_EXPLAINER_CALLS:-3}
export DATAMATE_EXPLAINER_CHECKS=${DATAMATE_EXPLAINER_CHECKS:-1}

curl -s -m 5 "${OLLAMA_BASE_URL%/v1}/api/version" > /dev/null || { echo "Ollama is not reachable at ${OLLAMA_BASE_URL%/v1} (start the app or: ollama serve)"; exit 1; }
if ! ollama list | awk 'NR>1{print $1}' | grep -qx "$OLLAMA_MODEL:latest\|$OLLAMA_MODEL"; then
  echo "creating $OLLAMA_MODEL from scripts/ollama/Modelfile"; ollama create "$OLLAMA_MODEL" -f scripts/ollama/Modelfile
fi
[ -f "docs/${SPLIT}_ids.txt" ] || { echo "no docs/${SPLIT}_ids.txt"; exit 1; }
[ -d "splits/$SPLIT" ] || python3 scripts/subsplit.py ids train "splits/$SPLIT" "docs/${SPLIT}_ids.txt"
[ -x .venv/bin/datamate ] || uv sync --frozen
echo "split=$SPLIT p2=${DATAMATE_P2:-fast} plan=${DATAMATE_PLAN_MODE:-spec} structure_cards=$DATAMATE_STRUCTURE_CARDS purchase_engine=$DATAMATE_PURCHASE_ENGINE explainer=$DATAMATE_EXPLAINER_CALLS/$DATAMATE_EXPLAINER_CHECKS plan_thinking=$DATAMATE_PLAN_THINKING model=$OLLAMA_MODEL"
exec .venv/bin/datamate run --backend ollama --split-dir "splits/$SPLIT" --name "$NAME" "$@"
