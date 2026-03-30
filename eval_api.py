"""
eval_api.py
-----------
Thin, import-friendly wrapper around query_pipeline for evaluation scripts.

Usage
-----
    from eval_api import query

    result = query("What is X?")
    result = query("What is X?", use_reranker=False)

The returned EvalResult dataclass contains everything you need for scoring:
    - result.answer          : str
    - result.t_retrieve_s    : float
    - result.t_rerank_s      : float   (0.0 when reranker is off)
    - result.t_generate_s    : float
    - result.t_total_s       : float
    - result.sources         : list[dict]   (file, page, bge_score, rerank_score)
    - result.chunks          : list[dict]   (full chunk dicts, useful for faithfulness checks)
    - result.n_retrieved      : int
    - result.n_final         : int
    - result.reranker_used   : bool
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from query_pipeline import ask          # single entry-point we'll keep stable


@dataclass
class EvalResult:
    answer: str
    t_retrieve_s: float
    t_rerank_s: float
    t_generate_s: float
    t_total_s: float
    n_retrieved: int
    n_final: int
    reranker_used: bool
    sources: list[dict] = field(default_factory=list)
    chunks: list[dict]  = field(default_factory=list)
    error: Optional[str] = None

    # ── Convenience ──────────────────────────────────────────────────────────
    def timing_summary(self) -> str:
        return (
            f"retrieve={self.t_retrieve_s:.2f}s  "
            f"rerank={self.t_rerank_s:.2f}s  "
            f"generate={self.t_generate_s:.2f}s  "
            f"total={self.t_total_s:.2f}s"
        )

    def to_dict(self) -> dict:
        return {
            "answer":        self.answer,
            "t_retrieve_s":  self.t_retrieve_s,
            "t_rerank_s":    self.t_rerank_s,
            "t_generate_s":  self.t_generate_s,
            "t_total_s":     self.t_total_s,
            "n_retrieved":   self.n_retrieved,
            "n_final":       self.n_final,
            "reranker_used": self.reranker_used,
            "sources":       self.sources,
            "error":         self.error,
        }


def query(
    question: str,
    use_reranker: bool = True,
    verbose: bool = False,
) -> EvalResult:
    """
    Run the full RAG pipeline and return a structured EvalResult.

    Parameters
    ----------
    question     : Natural-language question to send to the RAG system.
    use_reranker : When False the cross-encoder reranking step is skipped;
                   top_k_retrieve candidates are passed directly to the LLM.
    verbose      : Print pipeline timing to stdout (mirrors query_pipeline behaviour).

    Returns
    -------
    EvalResult dataclass.  On unexpected errors the .error field is set and
    .answer contains the error message so batch evals never crash mid-run.
    """
    try:
        answer, meta = ask(question, use_reranker=use_reranker, verbose=verbose)
        return EvalResult(
            answer       = answer,
            t_retrieve_s = meta.get("t_retrieve_s", 0.0),
            t_rerank_s   = meta.get("t_rerank_s",   0.0),
            t_generate_s = meta.get("t_generate_s", 0.0),
            t_total_s    = meta.get("t_total_s",    0.0),
            n_retrieved  = meta.get("n_candidates", 0),
            n_final      = meta.get("n_final",      0),
            reranker_used= meta.get("reranker_used", use_reranker),
            sources      = meta.get("sources", []),
            chunks       = meta.get("chunks",  []),
        )
    except Exception as exc:
        return EvalResult(
            answer        = f"ERROR: {exc}",
            t_retrieve_s  = 0.0,
            t_rerank_s    = 0.0,
            t_generate_s  = 0.0,
            t_total_s     = 0.0,
            n_retrieved   = 0,
            n_final       = 0,
            reranker_used = use_reranker,
            error         = str(exc),
        )