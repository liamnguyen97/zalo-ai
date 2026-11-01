#!/usr/bin/env bash
# Run the agents harness ON the GPU box against its local vLLM, resuming after crashes / vLLM restarts.
# usage: WORKERS=16 scripts/box_run.sh <split-dir> <run-name> <n_questions>
# An existing run folder with that name is resumed. Stop it (both the loop and the run, or the loop restarts it):
#   pkill -f "[b]ox_run.sh"; pkill -f "[d]atamate run"
cd "$(dirname "$0")/.."
export VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1} DATAMATE_BACKEND=vllm DATAMATE_MAX_RETRIES=8
W=${WORKERS:-16}; mkdir -p result/box_logs; OUT=result/box_logs/$2.out
echo "start $2 workers=$W $(date)" >> $OUT
RUN=$(ls -d result/runs/*_$2 2>/dev/null | tail -1)
if [ -z "$RUN" ]; then
  .venv/bin/datamate run --backend vllm --workers $W --split-dir $1 --runs-dir result/runs --name $2 >> $OUT 2>&1
fi
for a in $(seq 1 8); do
  RUN=$(ls -d result/runs/*_$2 | tail -1)
  [ "$(ls $RUN/logs | wc -l)" -ge "$3" ] && break
  until curl -s -m 5 ${VLLM_BASE_URL%/v1}/v1/models | grep -q Qwen; do sleep 30; done
  echo "resume $a $(date)" >> $OUT
  .venv/bin/datamate run --backend vllm --workers $W --split-dir $1 --runs-dir result/runs --resume-run $RUN >> $OUT 2>&1
done
echo "end $2 $(date)" >> $OUT
