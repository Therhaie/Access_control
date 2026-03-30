#!/bin/bash
# Starts the vLLM OpenAI-compatible server for Llama 3.1 70B
#
# Run this in a dedicated terminal or tmux pane BEFORE starting app.py.
# It keeps running and serves all requests from rag_engine.py and app.py.
#
# vLLM flags explained:
#   --dtype bfloat16        → half precision, best quality/VRAM tradeoff
#   --max-model-len 8192    → max context window (tokens); reduce if OOM
#   --gpu-memory-utilization 0.92  → leave ~8% VRAM free for overhead
#   --enable-prefix-caching → cache common system prompt KV across requests
#   --host 0.0.0.0          → accessible from the network (for Gradio etc.)
#   --port 8000             → default OpenAI-compatible port

source venv/bin/activate

# MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" 
MODEL="mistralai/Mistral-7B-Instruct-v0.2"

echo ""
echo "🚀 Starting vLLM server for ${MODEL}..."
echo "   API will be available at http://localhost:8000/v1"
echo "   Press Ctrl+C to stop."
echo ""

# vllm serve "$MODEL" \
#     --dtype bfloat16 \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.4 \
#     --enable-prefix-caching \
#     --host 0.0.0.0 \
#     --port 8000

vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.4 \
    --host 0.0.0.0 \
    --port 8000