"""
query_pipeline.py
-----------------
RAG query pipeline: retrieve → (optional rerank) → generate.

Changes vs original
~~~~~~~~~~~~~~~~~~~
* ask()           — accepts use_reranker=True/False; meta now includes
                    "reranker_used" so callers can tell what happened.
* ask_streaming() — same use_reranker flag threaded through.
* chat_loop()     — passes --no-rerank CLI flag down to ask_streaming().
* argparse block  — lets you drive everything from the command line:
                      python query_pipeline.py "my question"
                      python query_pipeline.py "my question" --no-rerank
                      python query_pipeline.py ingest [--reset]
                      python query_pipeline.py chat [--no-rerank]
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import CrossEncoder

from ingestion_pipeline import (
    CHROMA_PATH,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K_RERANK,
    TOP_K_RETRIEVE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    get_embedding_model,
    ingest_function,
)
from config import *          # COLLECTION and any other project-wide constants

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_PATH   = os.path.join(os.getcwd(), "./chroma_db")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL  = "BAAI/bge-reranker-large"

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. "
    "Answer using ONLY the context provided. "
    "If the answer is not in the context, say: "
    "'I don't have enough information in my knowledge base to answer that.' "
    "Be concise and cite the source filename when relevant."
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("Loading reranker model…")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_llm() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


# ── ChromaDB ──────────────────────────────────────────────────────────────────
def get_collection(reset: bool = False):
    print("ChromaDB path:", CHROMA_PATH)
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
        metadata={"hnsw:space": "cosine"},
    )


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k_retrieve: int = TOP_K_RETRIEVE,
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
        candidates.append(
            {
                "content":      doc,
                "source":       meta.get("triplet_index", "?"),
                "page":         meta.get("document_id",   "?"),
                "bge_score":    similarity,
                "rerank_score": None,
            }
        )

    return candidates, {
        "n_candidates": len(candidates),
        "t_retrieve_s": round(t_retrieve, 2),
    }


# ── Reranking ─────────────────────────────────────────────────────────────────
def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = TOP_K_RERANK,
) -> list[dict]:
    """Score candidate chunks with cross-encoder and keep the best top_k."""
    if not candidates:
        return candidates
    reranker = get_reranker()
    pairs    = [(query, c["content"]) for c in candidates]
    scores   = reranker.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = round(float(score), 4)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ── Prompt helpers ────────────────────────────────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source    = Path(c["source"]).name if isinstance(c["source"], str) else c["source"]
        bge       = c.get("bge_score", 0)
        rerank    = c.get("rerank_score")
        score_str = (
            f"bge={bge:.3f}, rerank={rerank:.3f}"
            if rerank is not None
            else f"bge={bge:.3f}"
        )
        parts.append(
            f"[{i}] source={source} | page={c['page']} | {score_str}\n{c['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


# ── Health check ──────────────────────────────────────────────────────────────
def _check_vllm_health() -> bool:
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


# ── Shared retrieve-and-optionally-rerank helper ──────────────────────────────
def _retrieve_and_rerank(
    question: str,
    use_reranker: bool,
) -> tuple[list[dict], dict]:
    """
    Retrieve candidates then, if use_reranker=True, score and trim with the
    cross-encoder.  Returns (chunks, meta) with timing keys already filled.
    """
    chunks, meta = retrieve(question)

    if use_reranker:
        t1              = time.time()
        chunks          = rerank(question, chunks)
        meta["t_rerank_s"] = round(time.time() - t1, 2)
    else:
        # Keep top_k_retrieve results ranked by BGE score only
        chunks = sorted(chunks, key=lambda x: x["bge_score"], reverse=True)[:TOP_K_RERANK]
        meta["t_rerank_s"] = 0.0

    meta["n_final"]       = len(chunks)
    meta["reranker_used"] = use_reranker
    return chunks, meta


# ── Public API ────────────────────────────────────────────────────────────────
def ask(
    question: str,
    use_reranker: bool = True,
    verbose: bool = True,
) -> tuple[str, dict]:
    """
    Non-streaming RAG query.

    Parameters
    ----------
    question     : The user's question.
    use_reranker : Set to False to skip cross-encoder reranking (faster,
                   useful for ablation studies in evaluation scripts).
    verbose      : Print timing summary to stdout.

    Returns
    -------
    (answer, meta)  where meta contains timing, source list, and chunk list.
    """
    t0 = time.time()
    chunks, meta = _retrieve_and_rerank(question, use_reranker)

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

    meta["t_generate_s"] = round(
        time.time() - t0 - meta["t_retrieve_s"] - meta["t_rerank_s"], 2
    )
    meta["t_total_s"] = round(time.time() - t0, 2)
    meta["sources"]   = [
        {
            "file":         str(c["source"]),
            "page":         c["page"],
            "bge_score":    round(c.get("bge_score")    or 0, 4),
            "rerank_score": round(c.get("rerank_score") or 0, 4),
        }
        for c in chunks
    ]
    meta["chunks"] = chunks   # kept for faithfulness / context-recall metrics

    if verbose:
        rerank_label = "✅" if use_reranker else "⏭ (skipped)"
        print(
            f"\n🔍 Retrieved {meta['n_candidates']} → final {meta['n_final']}  "
            f"reranker={rerank_label}"
        )
        print(
            f"⏱  retrieve={meta['t_retrieve_s']}s | "
            f"rerank={meta['t_rerank_s']}s | "
            f"generate={meta['t_generate_s']}s"
        )

    return answer, meta


def ask_streaming(
    question: str,
    use_reranker: bool = True,
) -> Generator[str, None, None]:
    """
    Streaming RAG query.  Yields text tokens then a final __META__ JSON chunk.
    """
    t0 = time.time()
    chunks, meta = _retrieve_and_rerank(question, use_reranker)

    if not chunks:
        yield "⚠ No documents found. Run ingestion first."
        return

    context  = build_context(chunks)
    messages = _messages(question, context)
    client   = get_llm()

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

    meta["t_generate_s"] = round(
        time.time() - t0 - meta["t_retrieve_s"] - meta["t_rerank_s"], 2
    )
    meta["t_total_s"] = round(time.time() - t0, 2)
    meta["sources"]   = [
        {
            "file":         str(c["source"]),
            "page":         c["page"],
            "bge_score":    round(c.get("bge_score")    or 0, 4),
            "rerank_score": round(c.get("rerank_score") or 0, 4),
        }
        for c in chunks
    ]
    yield "\n\n__META__" + json.dumps(meta)


# ── Interactive chat loop ─────────────────────────────────────────────────────
def chat_loop(use_reranker: bool = True) -> None:
    print("\n" + "═" * 64)
    print("  🧠  Local RAG  |  BGE-large + Cross-encoder + Llama 3.1")
    print(f"  LLM served by vLLM at {VLLM_BASE_URL}")
    reranker_status = "ON" if use_reranker else "OFF (--no-rerank)"
    print(f"  Reranker: {reranker_status}")
    print("  Commands: 'ingest', 'ingest --reset', 'quit'")
    print("═" * 64 + "\n")

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
        for token in ask_streaming(user_input, use_reranker=use_reranker):
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
                print(
                    f"   [{s['file']}  p.{s['page']}]  "
                    f"bge={s['bge_score']:.4f}  rerank={s['rerank_score']:.4f}"
                )
        if meta.get("t_total_s"):
            print(
                f"⏱  retrieve={meta.get('t_retrieve_s')}s  "
                f"rerank={meta.get('t_rerank_s')}s  "
                f"generate={meta.get('t_generate_s')}s  "
                f"total={meta.get('t_total_s')}s"
            )
        print()


# ── CLI entrypoint ────────────────────────────────────────────────────────────
def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="query_pipeline",
        description="Local RAG query pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- chat (default interactive loop) ------------------------------------
    chat_p = subparsers.add_parser("chat", help="Start interactive chat loop")
    chat_p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )

    # ---- ask (single question, prints answer + timing) ----------------------
    ask_p = subparsers.add_parser("ask", help="Ask a single question and exit")
    ask_p.add_argument("question", help="Question to ask")
    ask_p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    ask_p.add_argument(
        "--json",
        action="store_true",
        help="Dump full result as JSON (useful for eval scripts)",
    )

    # ---- ingest -------------------------------------------------------------
    ingest_p = subparsers.add_parser("ingest", help="Ingest documents into ChromaDB")
    ingest_p.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing collection before ingesting",
    )

    return parser


if __name__ == "__main__":
    import sys

    parser = _build_parser()
    # Default to 'chat' when no sub-command given (preserves old behaviour)
    args = parser.parse_args(sys.argv[1:] if sys.argv[1:] else ["chat"])

    if args.command == "ingest":
        ingest_function(reset=args.reset)

    elif args.command == "ask":
        use_reranker = not args.no_rerank
        answer, meta = ask(args.question, use_reranker=use_reranker, verbose=not args.json)
        if args.json:
            import dataclasses
            print(
                json.dumps(
                    {
                        "answer":        answer,
                        "t_retrieve_s":  meta.get("t_retrieve_s"),
                        "t_rerank_s":    meta.get("t_rerank_s"),
                        "t_generate_s":  meta.get("t_generate_s"),
                        "t_total_s":     meta.get("t_total_s"),
                        "reranker_used": meta.get("reranker_used"),
                        "sources":       meta.get("sources", []),
                    },
                    indent=2,
                )
            )
        else:
            print(f"\nAnswer: {answer}")
            if meta.get("sources"):
                print("\n📚 Sources:")
                for s in meta["sources"]:
                    print(f"   [{s['file']}  p.{s['page']}]  bge={s['bge_score']:.4f}")

    else:  # chat
        use_reranker = not args.no_rerank
        chat_loop(use_reranker=use_reranker)