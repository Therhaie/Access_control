"""
trace_eval.py
=============
TRACE RAG Evaluation Framework
--------------------------------
Implements the four TRACe metrics for RAG evaluation:

  T — context Relevance      (retriever)   Does the retrieved context actually address the query?
  R — answer faithfulness    (generator)   Is every claim in the answer grounded in the context?
  A — context utilization    (generator)   Did the LLM use the relevant parts of the context?
  C — answer Completeness    (generator)   Does the answer cover everything needed vs ground truth?

Plus two auxiliary metrics always recorded:
  • answer_correctness  — semantic / token overlap vs ground-truth answer (no judge call needed)
  • timing              — retrieve / rerank / generate latencies from the pipeline

Judge LLM
---------
All LLM-as-judge calls go to your local vLLM endpoint (port 8001 by default).
The judge is prompted to return ONLY a single integer: 1 (pass) or 0 (fail).
Final metric scores are the mean over all sample verdicts, giving a value in [0, 1].

Dataset format  (JSON list)
--------------
[
  {
    "question":       "What is the boiling point of water?",
    "ground_truth":   "100 degrees Celsius at sea level.",
    "document":       "Water boils at 100 °C (212 °F) under standard atmospheric pressure."
  },
  ...
]

Storage
-------
Every evaluation run is appended to  results/eval_runs.jsonl  (one JSON object per line).
This format is trivially loadable with pandas for plotting:

    import pandas as pd
    df = pd.read_json("results/eval_runs.jsonl", lines=True)

See the bottom of this file for a usage example and CLI entry point.

Usage examples
--------------
# Run a full evaluation (with reranker):
python trace_eval.py --dataset data/my_dataset.json

# Ablation: compare with/without reranker
python trace_eval.py --dataset data/my_dataset.json --no-rerank

# Limit to first 10 samples (quick smoke-test)
python trace_eval.py --dataset data/my_dataset.json --limit 10

# Custom judge endpoint / model
python trace_eval.py --dataset data/my_dataset.json \\
    --judge-url http://localhost:8002/v1/chat/completions \\
    --judge-model mistralai/Mistral-7B-Instruct-v0.3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── Import the pipeline API we built in eval_api.py ──────────────────────────
from eval_api import EvalResult, query

# ── Storage directory ─────────────────────────────────────────────────────────
RESULTS_DIR  = Path("results")
RUNS_FILE    = RESULTS_DIR / "eval_runs.jsonl"
SAMPLES_FILE = RESULTS_DIR / "eval_samples.jsonl"   # per-question detail

# ── Judge defaults ────────────────────────────────────────────────────────────
# DEFAULT_JUDGE_URL   = "http://localhost:8002/v1/chat/completions"
# DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

DEFAULT_JUDGE_URL   = "http://localhost:8000/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Judge LLM caller
# ═══════════════════════════════════════════════════════════════════════════════

def call_vllm_judge(
    prompt: str,
    judge_url: str  = DEFAULT_JUDGE_URL,
    model: str      = DEFAULT_JUDGE_MODEL,
    timeout: int    = 30,
) -> str:
    """
    Send a prompt to the vLLM judge and return the raw text response.
    Keeps max_tokens small — we only expect "0" or "1" back.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,   # deterministic judge
    }
    try:
        resp = requests.post(judge_url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"    ⚠  Judge call failed: {exc}")
        return "-1"   # sentinel → will be excluded from mean


def _parse_binary(raw: str) -> Optional[int]:
    """Extract 0 or 1 from judge response.  Returns None on parse failure."""
    # Look for the first digit in the response
    m = re.search(r"[01]", raw)
    if m:
        return int(m.group())
    return None


def judge_binary(
    prompt: str,
    judge_url: str = DEFAULT_JUDGE_URL,
    model: str     = DEFAULT_JUDGE_MODEL,
) -> Optional[int]:
    """Call judge and parse result as 0/1. Returns None if unparseable."""
    raw = call_vllm_judge(prompt, judge_url=judge_url, model=model)
    return _parse_binary(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRACe metric prompts
# ═══════════════════════════════════════════════════════════════════════════════

# ── T: Context Relevance ──────────────────────────────────────────────────────
# Retriever metric: does what was retrieved actually address the query?
CONTEXT_RELEVANCE_PROMPT = """\
You are a strict evaluation judge for a retrieval-augmented generation system.

TASK: Decide whether the retrieved context is relevant enough to answer the question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

RULES:
- Reply with 1 if the context contains information that is directly useful for answering the question.
- Reply with 0 if the context is off-topic, too vague, or completely missing the needed information.
- Reply with ONLY the digit 0 or 1. No explanation, no punctuation, nothing else.

VERDICT:"""


# ── R: Answer Faithfulness ────────────────────────────────────────────────────
# Generator metric: are all claims in the answer supported by the context?
ANSWER_FAITHFULNESS_PROMPT = """\
You are a strict evaluation judge for a retrieval-augmented generation system.

TASK: Decide whether every factual claim in the answer is supported by the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

RULES:
- Reply with 1 if ALL claims in the answer can be verified from the context (no hallucinations).
- Reply with 0 if ANY claim in the answer contradicts or goes beyond what the context states.
- Do not use your own knowledge — judge ONLY based on what the context says.
- Reply with ONLY the digit 0 or 1. No explanation, no punctuation, nothing else.

VERDICT:"""


# ── A: Context Utilization ────────────────────────────────────────────────────
# Generator metric: did the LLM actually use the relevant parts of the context?
CONTEXT_UTILIZATION_PROMPT = """\
You are a strict evaluation judge for a retrieval-augmented generation system.

TASK: Decide whether the answer makes good use of the relevant information in the context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

RULES:
- Reply with 1 if the answer incorporates the key relevant information from the context.
- Reply with 0 if the answer ignores important relevant information that was present in the context.
- Reply with ONLY the digit 0 or 1. No explanation, no punctuation, nothing else.

VERDICT:"""


# ── C: Answer Completeness ────────────────────────────────────────────────────
# Generator metric: does the answer cover all aspects compared to the ground truth?
ANSWER_COMPLETENESS_PROMPT = """\
You are a strict evaluation judge for a retrieval-augmented generation system.

TASK: Decide whether the generated answer is complete relative to the reference answer.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{ground_truth}

GENERATED ANSWER:
{answer}

RULES:
- Reply with 1 if the generated answer covers all the key points of the reference answer.
- Reply with 0 if the generated answer is missing important information from the reference answer.
- Minor wording differences are acceptable — focus on factual coverage, not style.
- Reply with ONLY the digit 0 or 1. No explanation, no punctuation, nothing else.

VERDICT:"""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Auxiliary: answer correctness (no judge call — deterministic token overlap)
# ═══════════════════════════════════════════════════════════════════════════════

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Unigram F1 between predicted and reference answer (lowercased, punctuation-stripped).
    Same approach used in SQuAD evaluation scripts.
    Returns a float in [0, 1].
    """
    def tokenize(s: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", s.lower()).split()

    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set   = set(gt_tokens)
    common   = pred_set & gt_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall    = len(common) / len(gt_set)
    return 2 * precision * recall / (precision + recall)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-sample evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    """All scores and metadata for a single dataset entry."""
    sample_id:           str
    run_id:              str
    question:            str
    ground_truth:        str
    document:            str             # reference document from dataset
    generated_answer:    str
    retrieved_context:   str             # concatenated context passed to judge
    # TRACe scores (0/1 binary, or None if judge failed)
    context_relevance:   Optional[int]
    answer_faithfulness: Optional[int]
    context_utilization: Optional[int]
    answer_completeness: Optional[int]
    # Auxiliary
    answer_correctness_f1: float         # token F1 vs ground truth
    # Timing
    t_retrieve_s:  float
    t_rerank_s:    float
    t_generate_s:  float
    t_total_s:     float
    # Config
    reranker_used: bool
    n_retrieved:   int
    n_final:       int
    # Bookkeeping
    timestamp:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error:         Optional[str] = None


def evaluate_sample(
    sample: dict,
    run_id: str,
    use_reranker: bool = True,
    judge_url: str     = DEFAULT_JUDGE_URL,
    judge_model: str   = DEFAULT_JUDGE_MODEL,
    verbose: bool      = True,
) -> SampleResult:
    """
    Run the full TRACE evaluation for a single dataset entry.

    Parameters
    ----------
    sample       : dict with keys 'question', 'ground_truth', 'document'
    run_id       : unique ID of the current evaluation run
    use_reranker : passed through to the RAG pipeline
    judge_url    : vLLM judge endpoint
    judge_model  : judge model name
    verbose      : print per-sample progress

    Returns
    -------
    SampleResult dataclass
    """
    question     = sample["question"]
    ground_truth = sample["ground_truth"]
    document     = sample.get("document", "")
    sample_id    = str(uuid.uuid4())

    if verbose:
        print(f"\n  Q: {question[:80]}{'…' if len(question) > 80 else ''}")

    # ── 1. Query the RAG pipeline ─────────────────────────────────────────────
    rag: EvalResult = query(question, use_reranker=use_reranker, verbose=False)

    if rag.error:
        return SampleResult(
            sample_id=sample_id, run_id=run_id,
            question=question, ground_truth=ground_truth, document=document,
            generated_answer="", retrieved_context="",
            context_relevance=None, answer_faithfulness=None,
            context_utilization=None, answer_completeness=None,
            answer_correctness_f1=0.0,
            t_retrieve_s=0.0, t_rerank_s=0.0,
            t_generate_s=0.0, t_total_s=0.0,
            reranker_used=use_reranker, n_retrieved=0, n_final=0,
            error=rag.error,
        )

    # Build a flat context string from retrieved chunks for the judge
    retrieved_context = "\n\n---\n\n".join(
        c.get("content", "") for c in rag.chunks
    )

    # ── 2. Run TRACe judges ───────────────────────────────────────────────────

    # T — Context Relevance
    if verbose:
        print("    → T: context relevance …", end=" ", flush=True)
    cr_score = judge_binary(
        CONTEXT_RELEVANCE_PROMPT.format(
            question=question,
            context=retrieved_context,
        ),
        judge_url=judge_url, model=judge_model,
    )
    if verbose:
        print(cr_score)

    # R — Answer Faithfulness
    if verbose:
        print("    → R: answer faithfulness …", end=" ", flush=True)
    af_score = judge_binary(
        ANSWER_FAITHFULNESS_PROMPT.format(
            context=retrieved_context,
            answer=rag.answer,
        ),
        judge_url=judge_url, model=judge_model,
    )
    if verbose:
        print(af_score)

    # A — Context Utilization
    if verbose:
        print("    → A: context utilization …", end=" ", flush=True)
    cu_score = judge_binary(
        CONTEXT_UTILIZATION_PROMPT.format(
            question=question,
            context=retrieved_context,
            answer=rag.answer,
        ),
        judge_url=judge_url, model=judge_model,
    )
    if verbose:
        print(cu_score)

    # C — Answer Completeness
    if verbose:
        print("    → C: answer completeness …", end=" ", flush=True)
    ac_score = judge_binary(
        ANSWER_COMPLETENESS_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            answer=rag.answer,
        ),
        judge_url=judge_url, model=judge_model,
    )
    if verbose:
        print(ac_score)

    # ── 3. Deterministic metrics ──────────────────────────────────────────────
    f1 = token_f1(rag.answer, ground_truth)
    if verbose:
        print(f"    → F1 (correctness): {f1:.3f}")
        print(f"    ⏱  {rag.timing_summary()}")

    return SampleResult(
        sample_id=sample_id,
        run_id=run_id,
        question=question,
        ground_truth=ground_truth,
        document=document,
        generated_answer=rag.answer,
        retrieved_context=retrieved_context,
        context_relevance=cr_score,
        answer_faithfulness=af_score,
        context_utilization=cu_score,
        answer_completeness=ac_score,
        answer_correctness_f1=f1,
        t_retrieve_s=rag.t_retrieve_s,
        t_rerank_s=rag.t_rerank_s,
        t_generate_s=rag.t_generate_s,
        t_total_s=rag.t_total_s,
        reranker_used=rag.reranker_used,
        n_retrieved=rag.n_retrieved,
        n_final=rag.n_final,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Run-level aggregation and storage
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunSummary:
    """Aggregated metrics for a complete evaluation run."""
    run_id:       str
    timestamp:    str
    dataset_path: str
    n_samples:    int
    n_errors:     int
    use_reranker: bool
    judge_model:  str
    judge_url:    str

    # TRACe means (None if all samples failed for that metric)
    context_relevance_mean:   Optional[float]
    answer_faithfulness_mean: Optional[float]
    context_utilization_mean: Optional[float]
    answer_completeness_mean: Optional[float]
    trace_mean:               Optional[float]   # mean of the four above

    # Auxiliary
    answer_correctness_f1_mean: float

    # Timing means
    t_retrieve_mean:  float
    t_rerank_mean:    float
    t_generate_mean:  float
    t_total_mean:     float


def _safe_mean(values: list[Optional[int | float]]) -> Optional[float]:
    """Mean ignoring None values.  Returns None if no valid values."""
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def aggregate(
    samples: list[SampleResult],
    run_id: str,
    dataset_path: str,
    use_reranker: bool,
    judge_model: str,
    judge_url: str,
) -> RunSummary:
    """Compute run-level means from per-sample results."""
    valid = [s for s in samples if s.error is None]

    cr   = _safe_mean([s.context_relevance   for s in valid])
    af   = _safe_mean([s.answer_faithfulness for s in valid])
    cu   = _safe_mean([s.context_utilization for s in valid])
    ac   = _safe_mean([s.answer_completeness for s in valid])

    trace_parts = [x for x in [cr, af, cu, ac] if x is not None]
    trace_mean  = round(sum(trace_parts) / len(trace_parts), 4) if trace_parts else None

    return RunSummary(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_path=str(dataset_path),
        n_samples=len(samples),
        n_errors=len(samples) - len(valid),
        use_reranker=use_reranker,
        judge_model=judge_model,
        judge_url=judge_url,
        context_relevance_mean=cr,
        answer_faithfulness_mean=af,
        context_utilization_mean=cu,
        answer_completeness_mean=ac,
        trace_mean=trace_mean,
        answer_correctness_f1_mean=_safe_mean(
            [s.answer_correctness_f1 for s in valid]
        ) or 0.0,
        t_retrieve_mean=_safe_mean([s.t_retrieve_s for s in valid]) or 0.0,
        t_rerank_mean  =_safe_mean([s.t_rerank_s   for s in valid]) or 0.0,
        t_generate_mean=_safe_mean([s.t_generate_s for s in valid]) or 0.0,
        t_total_mean   =_safe_mean([s.t_total_s    for s in valid]) or 0.0,
    )


# ── Storage helpers ───────────────────────────────────────────────────────────

def _append_jsonl(path: Path, obj: dict) -> None:
    """Append a single JSON object as a new line to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_sample(sample: SampleResult) -> None:
    _append_jsonl(SAMPLES_FILE, asdict(sample))


def save_run(summary: RunSummary) -> None:
    _append_jsonl(RUNS_FILE, asdict(summary))


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_summary(s: RunSummary) -> None:
    bar = "═" * 62
    print(f"\n{bar}")
    print(f"  TRACE Evaluation Run  [{s.run_id[:8]}]")
    print(f"  Dataset : {s.dataset_path}")
    print(f"  Samples : {s.n_samples}  (errors: {s.n_errors})")
    print(f"  Reranker: {'ON' if s.use_reranker else 'OFF'}")
    print(bar)
    print(f"  T  Context Relevance      : {s.context_relevance_mean   or 'N/A'}")
    print(f"  R  Answer Faithfulness    : {s.answer_faithfulness_mean or 'N/A'}")
    print(f"  A  Context Utilization    : {s.context_utilization_mean or 'N/A'}")
    print(f"  C  Answer Completeness    : {s.answer_completeness_mean or 'N/A'}")
    print(f"  {'─'*54}")
    print(f"     TRACe mean             : {s.trace_mean               or 'N/A'}")
    print(f"     Answer Correctness F1  : {s.answer_correctness_f1_mean:.4f}")
    print(bar)
    print(f"  ⏱  retrieve={s.t_retrieve_mean:.2f}s  "
          f"rerank={s.t_rerank_mean:.2f}s  "
          f"generate={s.t_generate_mean:.2f}s  "
          f"total={s.t_total_mean:.2f}s")
    print(f"{bar}\n")
    print(f"  Results saved to:")
    print(f"    {RUNS_FILE}     ← one line per run  (for trend plots)")
    print(f"    {SAMPLES_FILE}  ← one line per sample (for per-question analysis)")
    print(bar + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    dataset_path: str,
    use_reranker: bool = True,
    judge_url: str     = DEFAULT_JUDGE_URL,
    judge_model: str   = DEFAULT_JUDGE_MODEL,
    limit: Optional[int] = None,
    verbose: bool      = True,
) -> RunSummary:
    """
    Full TRACE evaluation over a JSON dataset.

    Parameters
    ----------
    dataset_path : path to JSON file (list of {question, ground_truth, document})
    use_reranker : toggle reranker in the RAG pipeline
    judge_url    : vLLM judge endpoint
    judge_model  : judge model identifier
    limit        : only evaluate the first N samples (useful for quick tests)
    verbose      : print per-sample progress

    Returns
    -------
    RunSummary dataclass (also persisted to disk)
    """
    run_id = str(uuid.uuid4())
    t_run_start = time.time()

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(dataset_path, encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)

    if limit:
        dataset = dataset[:limit]

    print(f"\n{'═'*62}")
    print(f"  TRACE Evaluation  run_id={run_id[:8]}")
    print(f"  Dataset: {dataset_path}  ({len(dataset)} samples)")
    print(f"  Reranker: {'ON' if use_reranker else 'OFF'}")
    print(f"  Judge:   {judge_model}")
    print(f"{'═'*62}")

    # ── Evaluate samples ──────────────────────────────────────────────────────
    results: list[SampleResult] = []
    for i, sample in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}]", end="")
        result = evaluate_sample(
            sample=sample,
            run_id=run_id,
            use_reranker=use_reranker,
            judge_url=judge_url,
            judge_model=judge_model,
            verbose=verbose,
        )
        results.append(result)
        save_sample(result)    # persist immediately (safe if run crashes mid-way)

    # ── Aggregate and save run summary ────────────────────────────────────────
    summary = aggregate(
        samples=results,
        run_id=run_id,
        dataset_path=dataset_path,
        use_reranker=use_reranker,
        judge_model=judge_model,
        judge_url=judge_url,
    )
    save_run(summary)
    print_summary(summary)
    print(f"  Total wall time: {time.time() - t_run_start:.1f}s")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trace_eval",
        description="TRACE RAG Evaluation — context Relevance, answer Faithfulness, "
                    "context Utilization, answer Completeness",
    )
    p.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to evaluation dataset JSON file",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking in the RAG pipeline",
    )
    p.add_argument(
        "--judge-url",
        default=DEFAULT_JUDGE_URL,
        help=f"vLLM judge endpoint (default: {DEFAULT_JUDGE_URL})",
    )
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model name (default: {DEFAULT_JUDGE_MODEL})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples (quick smoke-test)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample progress output",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    run_evaluation(
        dataset_path =args.dataset,
        use_reranker =not args.no_rerank,
        judge_url    =args.judge_url,
        judge_model  =args.judge_model,
        limit        =args.limit,
        verbose      =not args.quiet,
    )
