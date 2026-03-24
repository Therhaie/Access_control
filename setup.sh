



pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet


# ── [6/6] HuggingFace login ──────────────────────────────────────────────────
echo ""
echo "▶ [6/6] HuggingFace login..."
echo ""
echo "  Llama 3.1 is a GATED model — accept the license once at:"
echo "  → https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct"
echo ""
echo "  vLLM downloads the weights (~140GB) automatically on first"
echo "  'bash start_vllm.sh' — no manual download step needed."
echo ""
read -rp "  Run huggingface-cli login now? (y/n): " hf_login

if [[ "$hf_login" =~ ^[Yy]$ ]]; then
    huggingface-cli login
    echo "  ✅ HuggingFace credentials saved."
else
    echo "  ⏭  Skipped. Run before starting vLLM:"
    echo "     source ${VENV_DIR}/bin/activate && huggingface-cli login"
fi