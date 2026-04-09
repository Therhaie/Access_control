import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
import json
from pathlib import Path
import os
import re
from config import VLLM_EMBED_BASE_URL, VLLM_API_KEY, EMBED_MODEL
from config import *
from typing import List, Callable
from security import *

load_dotenv()

# ── Config
CHROMA_PATH    = "./chroma_db"
DOCS_PATH      = "./documents"
VLLM_BASE_URL  = "http://localhost:8000/v1"
VLLM_API_KEY   = "no-key-needed"
LLM_MODEL      = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"

CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
TOP_K_RETRIEVE = 500
TOP_K_RERANK   = 100

MAX_TOKENS     = 1024
TEMPERATURE    = 0.1

# ── Helpers
def parse_phrase_id(phrase_id: str) -> tuple[str, str]:
    """
    '0a'  → doc_id='0',  phrase_seq='a'
    '12c' → doc_id='12', phrase_seq='c'
    """
    match = re.match(r'^(\d+)([a-z]+)$', phrase_id)
    if not match:
        raise ValueError(f"Unexpected phrase_id format: '{phrase_id}'")
    return match.group(1), match.group(2)

def parse_chunk_type(raw_text: str) -> tuple[str, str]:
    """
    'Title: Emergent ...'  → ('title',   'Emergent ...')
    'Passage: Recent ...'  → ('passage', 'Recent ...')

    We KEEP the prefix in page_content because BGE was trained with it —
    stripping it slightly degrades retrieval quality.
    """

    if raw_text.startswith("Title:"):
        return raw_text          # keep full string with prefix
    elif raw_text.startswith("Passage:"):
        return raw_text        # keep full string with prefix
    else:
        return raw_text

# ── 0. Extract the wisefull information from the dataset
def extract_meaningfull_data(json_path, output_path=os.path.join(os.getcwd(),'documents_RAGBench', 'merged_id_triplets.json')):
    print(f"Loading dataset from: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
        print(type(doc))
        print(doc.keys())

    results = []
    for entry in doc.get('rows', []):
        row = entry.get('row', {})
        sentences = row.get('documents_sentences', [])
        question = row.get('question', '')
        response = row.get('response', '')
        id_triplet = row.get('id', '')

        results.append({
            "sentences": sentences,
            "question": question,
            "response": response,
            "id_triplets": id_triplet
        })

        

    with open(output_path, 'w') as f:
        json.dump(results, f)


def remove_duplicates_in_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        file = json.load(f)

    seen = set()
    unique_data = []
    for entry in file:
        sentences_ = []
        for document in entry["sentences"]:
            for sentence in document:
                # identifier =entry["row"]["documents_sentences"]
                if sentence[1] not in seen:
                    seen.add(sentence[1])
                    sentences_.append(sentence)
        unique_data.append(
            {
                "sentences": sentences_,
                "question": entry["question"],
                "response": entry["response"],
                "id_triplets":entry["id_triplets"]
            }
        )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2)





class CustomEmbeddings:
    """
    Drop-in replacement for OpenAIEmbeddings.
 
    Wraps the base model and applies:
      1. Optional rotation (orthogonal transform, same matrix every call)
      2. Optional extra dimension appending
      3. L2 normalisation (so cosine space stays valid)
 
    Both embed_documents() and embed_query() apply the *same* pipeline,
    which is essential — stored and query vectors must live in the same space.
 
    OUTPUT FORMAT
    -------------
    ChromaDB / LangChain expect List[List[float]].
    Internally we work with np.ndarray for speed, then convert at the end.
 
    Example
    -------
    >>> emb = CustomEmbeddings(rotate=True, extra_dims=4, extra_mode="zeros")
    >>> vecs = emb.embed_documents(["hello world", "foo bar"])
    >>> len(vecs[0])   # original_dim + 4
    """
 
    def __init__(
        self,
        base_model: OpenAIEmbeddings | None = None,
        rotate: bool = True,
        rotation_seed: int = 42,
        extra_dims: int = 0,          # set > 0 to append dimensions
        extra_mode: str = "zeros",    # "zeros" | "random" | "norm"
        normalize: bool = False,
    ):
        self._base = base_model or OpenAIEmbeddings(
            model=EMBED_MODEL,
            openai_api_base=VLLM_EMBED_BASE_URL,
            openai_api_key=VLLM_API_KEY,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )
        self.rotate       = rotate
        self.rotation_seed = rotation_seed
        self.extra_dims   = extra_dims
        self.extra_mode   = extra_mode
        self.normalize    = normalize
 
        # Rotation matrix is built lazily once we know the embedding dimension.
        self._R: np.ndarray | None = None
 
    # ── internal helpers ──────────────────────────────────────────────────────
 
    def _get_rotation(self, dim: int) -> np.ndarray:
        if self._R is None or self._R.shape[0] != dim:
            self._R = make_rotation_matrix(dim, seed=self.rotation_seed)
        return self._R
 
    def _postprocess(self, raw: List[List[float]]) -> List[List[float]]:
        """Apply the full post-processing pipeline and return List[List[float]]."""
        vecs = np.array(raw, dtype=np.float32)   # (N, dim)
 
        if self.rotate:
            R    = self._get_rotation(vecs.shape[1])
            vecs = rotate_vectors(vecs, R)
 
        if self.extra_dims > 0:
            vecs = append_extra_dimensions(vecs, self.extra_dims, self.extra_mode)
 
        if self.normalize:
            vecs = l2_normalize(vecs)
 
        return vecs.tolist()
 
    # ── LangChain Embeddings interface ────────────────────────────────────────
 
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
      
        raw = self._base.embed_documents(texts)
        return self._postprocess(raw)
 
    def embed_query(self, text: str) -> List[float]:
        raw   = self._base.embed_query(text)        # List[float]
        result = self._postprocess([raw])            # → List[List[float]]
        return result[0]                             # → List[float]


# ── 1. Load and parse
def load_file(json_path: Path) -> list[Document]:
    """
    Reads the JSON triplet dataset and returns a flat list of LangChain Documents.
    Each Document corresponds to one phrase (title or passage).

    Metadata stored per chunk:
        - document_id  : str  — the numeric part of the phrase id (e.g. '0', '12')
        - chunk_type   : str  — 'title' | 'passage' | 'unknown'
        - phrase_seq   : str  — alphabetic sequence within the document (e.g. 'a', 'b')
        - triplet_index: int  — position of the parent triplet in the dataset
    """

    with open(json_path, "r", encoding="utf-8") as f:    
    # with open(os.path.join(os.getcwd(),'documents_RAGBench', 'merged_id_triplets.json'), 'r') as f:
        triplets = json.load(f)

    print(f"  → {len(triplets)} triplets found")

    documents: list[Document] = []
    skipped = 0

    for triplet_idx, triplet in enumerate(triplets):
        # ── Safely unpack the triplet
        question  = triplet.get("question", "")
        response    = triplet.get("response", "")
        sentences   = triplet.get("sentences", [[]])
        id_triplet = triplet.get("id_triplets", "")

        # ── Each phrase is a [phrase_id, raw_text] pair
        for phrase in sentences:
            # for phrase in doc:

            phrase_id, raw_text = phrase

            try:
                doc_id, phrase_seq = parse_phrase_id(phrase_id)
            except ValueError:
                skipped += 1
                continue

            # Chroma only accepts str / int / float / bool in metadata
            if raw_text.startswith("Title:"):
                page_content = raw_text.replace("Title: ", "")
            elif raw_text.startswith("Passage:"):
                page_content = raw_text.replace("Passage: ", "")
            else:
                page_content = raw_text
            
            documents.append(Document(
                page_content=page_content,
                metadata={
                    "document_id"  : doc_id,        # str  e.g. '0'
                    "phrase_seq"   : phrase_seq,     # str  e.g. 'a'
                    "triplet_index": id_triplet,    # int  for tracing back to source
                }
            ))


    print(f"  → {len(documents)} chunks loaded  |  {skipped} entries skipped")

    # Debug preview
    # for doc in documents[:3]:
    #     print(f"\n  [{doc.metadata['document_id']}{doc.metadata['phrase_seq']}] "
    #           f"({doc.metadata['chunk_type']}) {doc.page_content[:80]}…")

    return documents


def get_classical_embedding():
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_base=VLLM_EMBED_BASE_URL,  # points to :8001
        openai_api_key=VLLM_API_KEY,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False 
    )

def get_embedding_model_rotate():
    """
    Return a custom embedding model using a wrapper to apply either a rotation or the augmentation of dimension to the embedded data before they are added to the database.
    """
    return CustomEmbeddings(
        base_model=get_classical_embedding(),
        rotate=False,
        rotation_seed=42,
        extra_dims=0,
        extra_mode=0,
        normalize=False
    )

def get_embedding_model():
    if COLLECTION == "rotated_db":
        return get_embedding_model_rotate()
    else:
        return get_classical_embedding()

# ── 4. Build / update ChromaDB vector store
def create_vector_store(chunks, persist_directory=CHROMA_PATH):
    print("\nCreating embeddings and storing in ChromaDB …")

    embedding_model = get_embedding_model()

    # When custom rotation for each attribut / role add a function to modify the embedding model used

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},   # cosine suits BGE
    )

    print(f"  → Vector store saved to: {persist_directory}")
    print(f"  → Total vectors: {vectorstore._collection.count()}")

    return vectorstore

def ingest_function(path):
    path = os.path.join(os.getcwd(), 'documents_RAGBench', 'merged_id_triplets.json')
    # extract_meaningfull_data(path)
    chunks  = load_file(path)
    vectore_store = create_vector_store(chunks)
    print("\n ingestion completed")



def main():
    path_raw_data = os.path.join(os.getcwd(), 'documents_RAGBench', 'data.json')
    path_merged_id_triplets = os.path.join(os.getcwd(), 'documents_RAGBench', 'merged_id_triplets2.json')
    
    extract_meaningfull_data(path_raw_data, path_merged_id_triplets)
    # path_output = os.path.join(os.getcwd(),'documents_RAGBench', 'merged_id_triplets_no_duplicates.json')   
    # remove_duplicates_in_json(path, path_output)
    # chunks  = load_file(path_output)
    # vectore_store = create_vector_store(chunks)
    print("\n ingestion completed")

if __name__ == "__main__":
    main()



