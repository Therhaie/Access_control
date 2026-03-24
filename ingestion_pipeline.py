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

# region
# # ── 1. Load documents
# def load_file(doc_path: str):
#     print(f"Loading documents from: {doc_path}")

#     if not os.path.exists(doc_path):
#         raise FileNotFoundError(f"Directory not found: {doc_path}")

#     loader = DirectoryLoader(path=doc_path, glob="*.txt", loader_cls=TextLoader)
#     documents = loader.load()

#     if not documents:
#         raise FileNotFoundError(f"No .txt files found in {doc_path}")

#     for i, doc in enumerate(documents[:1]):
#         print(f"\n Document {i+1}")
#         print(f"  Source  : {doc.metadata['source']}")
#         print(f"  Length  : {len(doc.page_content)} characters")
#         print(f"  Preview : {doc.page_content[:100]}")

#     return documents


# # ── 2. Split into chunks
# def split_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     print(f"\nSplitting into chunks (size={chunk_size}, overlap={overlap})")

#     splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
#     chunks = splitter.split_documents(documents)

#     print(f"  → {len(chunks)} chunks created")

#     for i, chunk in enumerate(chunks[:3]):
#         print(f"\n  Chunk {i+1} | Source: {chunk.metadata['source']}")
#         print(f"  {chunk.page_content[:120]}")
#         print("-" * 50)

#     return chunks
# endregion

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
def extract_meaningfull_data(json_path):
    print(f"Loading dataset from: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
        print(type(doc))
        print(doc.keys())
        results = [
    {
        "sentences": entry["row"]["documents_sentences"],
        "question": entry["row"]["question"],
        "response": entry["row"]["response"],
        "id_triplets":entry["row"]["id"]
    }
    for entry in doc["rows"]
    ]

    with open(os.path.join(os.getcwd(),'documents_RAGBench', 'merged_id_triplets.json'), 'w') as f:
        json.dump(results, f)



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
        normalize: bool = True,
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

        
    with open(os.path.join(os.getcwd(),'documents_RAGBench', 'merged_id_triplets.json'), 'r') as f:
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
        # print(sentences)
        # break


        # ── Each phrase is a [phrase_id, raw_text] pair
        for doc in sentences:
            for phrase in doc:

                phrase_id, raw_text = phrase

                try:
                    doc_id, phrase_seq = parse_phrase_id(phrase_id)
                except ValueError:
                    skipped += 1
                    continue
                # page_content = raw_text
                # chunk_type, page_content = parse_chunk_type(raw_text)

                # Chroma only accepts str / int / float / bool in metadata
                # if ("Title: " in raw_text): print("Found in raw_text")
                
                page_content = raw_text.replace("Title: ", "")
                
                # if ("Title: " in page_content): print("Found in page") 
                # else : print(page_content)
                
                page_content = raw_text.replace("Passage: ", "")
                # print(page_content)
                
                documents.append(Document(
                    page_content=page_content,
                    metadata={
                        "document_id"  : doc_id,        # str  e.g. '0'
                        "phrase_seq"   : phrase_seq,     # str  e.g. 'a'
                        "triplet_index": id_triplet,    # int  for tracing back to source
                    }
                ))


        # for phrase in sentences:
        #     # if not isinstance(phrase, list) or len(phrase) != 2:
        #     #     skipped += 1
        #     #     continue

        #     phrase_id, raw_text = phrase[0]
        #     # print(type(phrase_id), phrase_id)

        #     # print(type(phrase), phrase)
        #     # print(type(raw_text), raw_text)
        #     # Remove Title: and Passage: at the beginning
        #     # if (raw_text.startswith("Title:") or raw_text.startswith("Passage:")):
                
        #     #     # print(repr(raw_text))
        #     #     page_content = raw_text.replace("Title: ", "")
        #     #     page_content = raw_text.replace("Passage: ", "")

        #     try:
        #         doc_id, phrase_seq = parse_phrase_id(phrase_id)
        #     except ValueError:
        #         skipped += 1
        #         continue
        #     # page_content = raw_text
        #     # chunk_type, page_content = parse_chunk_type(raw_text)

        #     # Chroma only accepts str / int / float / bool in metadata
        #     if ("Title: " in raw_text): print("Found in raw_text")
        #     page_content = raw_text.replace("Title: ", "")
        #     if ("Title: " in page_content): print("Found in page")
        #     page_content = raw_text.replace("Passage: ", "")
        #     print(page_content)
        #     documents.append(Document(
        #         page_content=page_content,
        #         metadata={
        #             "document_id"  : doc_id,        # str  e.g. '0'
        #             "phrase_seq"   : phrase_seq,     # str  e.g. 'a'
        #             "triplet_index": id_triplet,    # int  for tracing back to source
        #         }
        #     ))

    print(f"  → {len(documents)} chunks loaded  |  {skipped} entries skipped")

    # Debug preview
    # for doc in documents[:3]:
    #     print(f"\n  [{doc.metadata['document_id']}{doc.metadata['phrase_seq']}] "
    #           f"({doc.metadata['chunk_type']}) {doc.page_content[:80]}…")

    return documents




def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_base=VLLM_EMBED_BASE_URL,  # points to :8001
        openai_api_key=VLLM_API_KEY,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False 
    )

# ── 4. Build / update ChromaDB vector store
def create_vector_store(chunks, persist_directory=CHROMA_PATH):
    print("\nCreating embeddings and storing in ChromaDB …")

    embedding_model = get_embedding_model()

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
    path = os.path.join(os.getcwd(), 'documents_RAGBench', 'data.json')
    # extract_meaningfull_data(path)
    chunks  = load_file(path)
    vectore_store = create_vector_store(chunks)
    print("\n ingestion completed")




# region
# # ── 1. Load and parse
# def load_file(json_path: Path) -> list[Document]:
#     """
#     Reads the JSON triplet dataset and returns a flat list of LangChain Documents.
#     Each Document corresponds to one phrase (title or passage).

#     Metadata stored per chunk:
#         - document_id  : str  — the numeric part of the phrase id (e.g. '0', '12')
#         - chunk_type   : str  — 'title' | 'passage' | 'unknown'
#         - phrase_seq   : str  — alphabetic sequence within the document (e.g. 'a', 'b')
#         - triplet_index: int  — position of the parent triplet in the dataset
#     """

        
#     with open(os.path.join(os.getcwd(),'documents_RAGBench', 'merged.json'), 'r') as f:
#         triplets = json.load(f)

#     print(f"  → {len(triplets)} triplets found")

#     documents: list[Document] = []
#     skipped = 0

#     for triplet_idx, triplet in enumerate(triplets):
#         # ── Safely unpack the triplet
#         question  = triplet.get("question", "")
#         response    = triplet.get("response", "")
#         sentences   = triplet.get("sentences", [])
#         # print(sentences)
#         # break


#         # ── Each phrase is a [phrase_id, raw_text] pair
#         for phrase in sentences:
#             if not isinstance(phrase, list) or len(phrase) != 2:
#                 skipped += 1
#                 continue

#             phrase_id, raw_text = phrase

#             # Skip anything that's not a title or passage
#             if not (raw_text.startswith("Title:") or raw_text.startswith("Passage:")):
#                 skipped += 1
#                 continue

#             try:
#                 doc_id, phrase_seq = parse_phrase_id(phrase_id)
#             except ValueError:
#                 skipped += 1
#                 continue

#             chunk_type, page_content = parse_chunk_type(raw_text)

#             # Chroma only accepts str / int / float / bool in metadata
#             documents.append(Document(
#                 page_content=page_content,
#                 metadata={
#                     "document_id"  : doc_id,        # str  e.g. '0'
#                     "phrase_seq"   : phrase_seq,     # str  e.g. 'a'
#                     "chunk_type"   : chunk_type,     # str  'title' | 'passage'
#                     "triplet_index": triplet_idx,    # int  for tracing back to source
#                 }
#             ))

#     print(f"  → {len(documents)} chunks loaded  |  {skipped} entries skipped")

#     # # Debug preview
#     # for doc in documents[:3]:
#     #     print(f"\n  [{doc.metadata['document_id']}{doc.metadata['phrase_seq']}] "
#     #           f"({doc.metadata['chunk_type']}) {doc.page_content[:80]}…")

#     # return documents






# # ── 3. Embeddings via vLLM (OpenAI-compatible endpoint)
# def get_embedding_model():
#     """
#     vLLM exposes an OpenAI-compatible /v1/embeddings endpoint.
#     We point OpenAIEmbeddings at it and swap in BAAI/bge-large-en-v1.5.
#     Make sure you started a *separate* vLLM embedding server, or use a
#     dedicated embedding endpoint (see architecture notes below).
#     """
#     return OpenAIEmbeddings(
#         model=EMBED_MODEL,
#         openai_api_base=VLLM_BASE_URL,
#         openai_api_key=VLLM_API_KEY,
#         # BGE models benefit from this query prefix at retrieval time
#         # (handled separately in the query pipeline)
#     )


# # ── 4. Build / update ChromaDB vector store
# def create_vector_store(chunks, persist_directory=CHROMA_PATH):
#     print("\nCreating embeddings and storing in ChromaDB …")

#     embedding_model = get_embedding_model()

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         persist_directory=persist_directory,
#         collection_metadata={"hnsw:space": "cosine"},   # cosine suits BGE
#     )

#     print(f"  → Vector store saved to: {persist_directory}")
#     print(f"  → Total vectors: {vectorstore._collection.count()}")

#     return vectorstore


# # ── 5. (Optional) Incremental upsert – avoids re-embedding unchanged docs
# def upsert_vector_store(chunks, persist_directory=CHROMA_PATH):
#     """
#     Load an existing store and add only new chunks.
#     Useful for production pipelines where docs are updated incrementally.
#     """
#     print("\nUpserting chunks into existing ChromaDB …")

#     embedding_model = get_embedding_model()

#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embedding_model,
#         collection_metadata={"hnsw:space": "cosine"},
#     )
#     vectorstore.add_documents(chunks)

#     print(f"  → Total vectors after upsert: {vectorstore._collection.count()}")
#     return vectorstore


# # ── Main
# def main():
#     documents  = load_file(doc_path=DOCS_PATH)
#     chunks     = split_documents(documents)
#     vectorstore = create_vector_store(chunks)
#     print("\n✅ Ingestion complete.")
# endregion



def main():
    path = os.path.join(os.getcwd(), 'documents_RAGBench', 'data.json')
    # extract_meaningfull_data(path)
    chunks  = load_file(path)
    vectore_store = create_vector_store(chunks)
    print("\n ingestion completed")

if __name__ == "__main__":
    main()



