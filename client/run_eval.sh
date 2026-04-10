#!/usr/bin/bash

source ./.venv/bin/activate

export QWEN_BASE_URL="http://10.1.3.100:25541/v1"
export QWEN_API_KEY=
export QWEN_MODEL="qwen3.5-9b"

export OPENAI_BASE_URL="https://api.cphone.vip/v1"
export OPENAI_API_KEY=""
# export OPENAI_MODEL="gpt-4o"
export OPENAI_MODEL="gpt-5.4"

export GEMINI_BASE_URL="https://api.cphone.vip/v1"
export GEMINI_API_KEY=""
export GEMINI_MODEL="gemini-3.1-pro-preview"

# python evaluate_vlm_from_start.py --max-samples 1 --models qwen
python evaluate_vlm_from_start.py --max-samples 30 --models chatgpt
# python evaluate_vlm_from_start.py --max-samples 0 --models gemini
