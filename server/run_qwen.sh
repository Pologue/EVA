#! /usr/bin/bash

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/data/private/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype half \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768