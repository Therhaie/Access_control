
source venv/bin/activate

MODEL="BAAI/bge-large-en-v1.5"


echo ""
echo "🚀 Starting vLLM server for ${MODEL}..."
echo "   API will be available at http://localhost:8001/v1"
echo "   Press Ctrl+C to stop."
echo ""

echo "Starting embedding server on :8001..."



# vllm serve "$MODEL" \
#     --dtype bfloat16 \
#     --max-model-len 512 \
#     --gpu-memory-utilization 0.3 \
#     --enable-prefix-caching \
#     --host 0.0.0.0 \
#     --port 8001

vllm serve BAAI/bge-large-en-v1.5\
    --gpu-memory-utilization 0.1 \
    --host 0.0.0.0 \
    --port 8001