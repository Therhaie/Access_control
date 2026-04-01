# config.py  — single source of truth for both pipelines
import os
# ── Config

# CHROMA_PATH    = "./chroma_db"
CHROMA_PATH = os.path.join(os.getcwd(), './chroma_db')
DOCS_PATH      = "./documents"
COLLECTION     = 'baseline_db'
# COLLECTION = "my_knowledge_base" # Name classical embedded database
# COLLECTION = "rotated_db" # database with a rotation added to the embedding

# ── LLM server (port 8000)
VLLM_LLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY      = "no-key-needed"
LLM_MODEL         = "mistralai/Mistral-7B-Instruct-v0.2"

# ── Embedding server (port 8001)
VLLM_EMBED_BASE_URL = "http://localhost:8001/v1"
EMBED_MODEL         = "BAAI/bge-large-en-v1.5"

# ── Paths
CHROMA_PATH = "./chroma_db"
DOCS_PATH   = "./data/dataset.json"

# ── Retrieval
TOP_K_RETRIEVE = 20
TOP_K_RERANK   = 6
MAX_TOKENS     = 1024
TEMPERATURE    = 0.1
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
