#! /usr/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/data/private/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype half \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768