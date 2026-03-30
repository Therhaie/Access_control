# source .venv/bin/activate

source venv/bin/activate
# MODEL="mistral:7b-instruct-q4_K_M"

##### to launch mistral as a judge
MODEL="mistralai/Mistral-7B-Instruct-v0.3"

########## to launch DeBerta as a judge and do NLI
# MODEL="microsoft/DeBERTa-large"
# MODEL="microsoft/mdeberta-v3-base"



echo ""
echo "🚀 Starting vLLM server for ${MODEL}..."
echo "   API will be available at http://localhost:8002/v1"
echo "   Press Ctrl+C to stop."
echo ""

vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.4 \
    --host 0.0.0.0 \
    --port 8001


