#!/usr/bin/env bash
# Self-host Qwen3.5-9B with vLLM: one replica per GPU, OpenAI-compatible on ports 8000, 8001.
# Needs CUDA >= 12.8 (RTX 5090 / sm_120) and a vLLM build that supports Qwen3.5.
# For rerun determinism: VLLM_BATCH_INVARIANT=1 (beta) plus temperature 0.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
GPUS="${GPUS:-1}"                       # number of replicas / GPUs
BASE_PORT="${BASE_PORT:-8000}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"

for ((i = 0; i < GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}" \
  vllm serve "$MODEL" \
    --port $((BASE_PORT + i)) \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --seed 0 &
done
wait
