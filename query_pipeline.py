# query_pipeline.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
from ingestion_pipeline import get_embedding_model
import chromadb
from chromadb.config import Settings
# Re-use the same config constants
from typing import Generator
import json

from ingestion_pipeline import (
    CHROMA_PATH, VLLM_BASE_URL, VLLM_API_KEY,
    LLM_MODEL, EMBED_MODEL,
    TOP_K_RETRIEVE, TOP_K_RERANK,
    MAX_TOKENS, TEMPERATURE,
)
from config import *
# from config import CHROMA_PATH
from pathlib import Path
from ingestion_pipeline import ingest_function
from openai import OpenAI

CHROMA_PATH = os.path.join(os.getcwd(), './chroma_db')

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""




def get_retriever():
    # embedding_model = OpenAIEmbeddings(
    #     model=EMBED_MODEL,
    #     openai_api_base=VLLM_BASE_URL,
    #     openai_api_key=VLLM_API_KEY,
    # )
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # BGE models expect a task prefix on queries (not on stored passages)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K_RETRIEVE,
               # injected at query time only
        },
    )



def get_llm():
    return OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
    )




# def build_rag_chain():
#     retriever = get_retriever()
#     llm       = get_llm()
#     prompt    = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

#     def format_docs(docs):
#         return "\n\n---\n\n".join(
#             f"[Source: {d.metadata.get('source','?')}]\n{d.page_content}"
#             for d in docs
#         )

#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain


# def query(question: str) -> str:
#     chain = build_rag_chain()
#     return chain.invoke(question)

##################################################################

def _check_vllm_health():
    """Friendly error if vLLM server isn't running."""
    import httpx
    try:
        r = httpx.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=3)
        if r.status_code == 200:
            print("✅ vLLM server is running.")
            return True
    except Exception:
        pass
    print("❌ vLLM server not reachable at", VLLM_BASE_URL)
    print("   Start it with:  vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16")
    return False



# ── ChromaDB ──────────────────────────────────────────────────────────────────
def get_collection(reset: bool = False):
    print("path to the database given :", CHROMA_PATH)
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print("🗑  Cleared existing collection.")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # inner product on normalised vecs = cosine
    )


def retrieve(query: str,
    top_k_retrieve: int = TOP_K_RETRIEVE
    ) -> tuple[list[dict], dict]:
    
    t0         = time.time()
    collection = get_collection()
    embedder   = get_embedding_model()
    query_vec  = embedder.embed_query(BGE_QUERY_PREFIX + query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k_retrieve, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    t_retrieve = time.time() - t0

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1.0 - float(dist), 4)
        print("similarity",similarity)
        candidates.append({
            "content":      doc,
            "source":       meta.get("triplet_index", "?"),
            "page":         meta.get("document_id", "?"),
            "bge_score":    similarity,
            "rerank_score": None,
        })
    
 
        
    return candidates, {
    "n_candidates": len(candidates),
    "t_retrieve_s": round(t_retrieve, 2)
}

# ── Prompt helpers ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. "
    "Answer using ONLY the context provided. "
    "If the answer is not in the context, say: "
    "'I don't have enough information in my knowledge base to answer that.' "
    "Be concise and cite the source filename when relevant."
)

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = Path(c["source"]).name
        score  = c.get("rerank_score") or c.get("bge_score", 0)
        parts.append(
            f"[{i}] {source} (page {c['page']}, score {score:.3f})\n{c['content']}"
        )
    return "\n\n---\n\n".join(parts)

def _messages(question: str, context: str) -> list[dict]:
    """Build the chat messages list for the vLLM /v1/chat/completions call."""
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

def ask_streaming(question: str) -> Generator[str, None, None]:
    """
    Retrieve → rerank → stream answer tokens from vLLM.
    Yields text tokens, then a final __META__ JSON sentinel.

    vLLM handles concurrent requests via PagedAttention + continuous batching,
    so multiple Gradio users can stream simultaneously without blocking.
    """
    t0 = time.time()
    chunks, pipe_meta = retrieve(question)
    if not chunks:
        yield "⚠ No documents found. Run ingestion first."
        return

    context  = build_context(chunks)
    messages = _messages(question, context)
    client   = get_llm()

    # vLLM streaming via SSE — same interface as OpenAI
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

    # pipe_meta["t_generate_s"] = round(time.time() - t0 - pipe_meta["t_retrieve_s"] - pipe_meta["t_rerank_s"], 2)
    pipe_meta["t_generate_s"] = round(time.time() - t0 - pipe_meta["t_retrieve_s"], 2)

    pipe_meta["t_total_s"]    = round(time.time() - t0, 2)
    pipe_meta["sources"]      = [
        {"file": Path(c["source"]).name, "page": c["page"], "score": round(c.get("rerank_score") or 0, 4)}
        for c in chunks
    ]
    yield "\n\n__META__" + json.dumps(pipe_meta)


def ask(question: str, verbose: bool = True) -> tuple[str, dict]:
    """Non-streaming version — used by the batch evaluator."""
    t0 = time.time()
    chunks, pipe_meta = retrieve(question)
    if not chunks:
        return "⚠ No documents found.", {}

    context  = build_context(chunks)
    messages = _messages(question, context)
    client   = get_llm()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    answer = response.choices[0].message.content.strip()

    pipe_meta["t_generate_s"] = round(time.time() - t0 - pipe_meta["t_retrieve_s"] - pipe_meta["t_rerank_s"], 2)
    pipe_meta["t_total_s"]    = round(time.time() - t0, 2)
    pipe_meta["sources"]      = [
        {"file": Path(c["source"]).name, "page": c["page"], "score": round(c.get("rerank_score") or 0, 4)}
        for c in chunks
    ]
    pipe_meta["chunks"] = chunks

    if verbose:
        print(f"\n🔍 Retrieved {pipe_meta['n_candidates']} → reranked to {pipe_meta['n_final']}")
        print(f"⏱  retrieve={pipe_meta['t_retrieve_s']}s | rerank={pipe_meta['t_rerank_s']}s | generate={pipe_meta['t_generate_s']}s\n")

    return answer, pipe_meta



def chat_loop():
    print("\n" + "═"*64)
    print("  🧠  Local RAG  |  BGE-large + Cross-encoder + Llama 3.1 70B")
    print("  LLM served by vLLM at", VLLM_BASE_URL)
    print("  Commands: 'ingest', 'ingest --reset', 'quit'")
    print("═"*64 + "\n")

    _check_vllm_health()

    try:
        col   = get_collection()
        count = col.count()
        if count == 0:
            print("\n⚠  Knowledge base is empty.")
            print("   Drop files into ./documents/ then type 'ingest'\n")
        else:
            print(f"\n📦 Knowledge base ready — {count} chunks indexed.\n")
    except Exception as e:
        print(f"⚠  ChromaDB error: {e}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower().startswith("ingest"):
            ingest_function(reset="--reset" in user_input)
            continue

        print("\nAssistant: ", end="", flush=True)
        meta = {}
        for token in ask_streaming(user_input):
            if token.startswith("\n\n__META__"):
                try:
                    meta = json.loads(token.replace("\n\n__META__", ""))
                except Exception:
                    pass
            else:
                print(token, end="", flush=True)
        print()

        if meta.get("sources"):
            print("\n📚 Sources:")
            for s in meta["sources"]:
                print(f"   [{s['score']:.4f}] {s['file']}  p.{s['page']}")
        if meta.get("t_total_s"):
            print(
                f"⏱  retrieve={meta.get('t_retrieve_s')}s  "
                f"rerank={meta.get('t_rerank_s')}s  "
                f"generate={meta.get('t_generate_s')}s  "
                f"total={meta.get('t_total_s')}s"
            )
        print()



if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "ingest":
        ingest_function(reset="--reset" in args)
    else:
        chat_loop()
    # q = "What is the main topic of the documents?"
    # print(query(q))