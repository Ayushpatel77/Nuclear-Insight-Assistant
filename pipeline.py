"""
Pipeline Manager — Orchestrates the 4-agent pipeline.
Linear flow: Retriever → Explainer → Simplifier → Critic → Output
Each agent runs ONCE. No recursion. Fallback on failure.
"""

from typing import Dict, Callable, Optional

from agents.retriever_agent import retrieve, format_context
from agents.llm_agents import explainer_agent, simplifier_agent, critic_agent


def run_pipeline(
    query: str,
    top_k: int = 6,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Run the full multi-agent pipeline for a user query.

    Returns a dict with:
        - query
        - retrieved_chunks (list)
        - technical_answer
        - simple_answer
        - final_answer (critic output)
        - sources (list of unique source files)
        - pipeline_log (list of step messages)
        - error (str or None)
    """

    result = {
        "query":            query,
        "retrieved_chunks": [],
        "technical_answer": "",
        "simple_answer":    "",
        "final_answer":     "",
        "sources":          [],
        "pipeline_log":     [],
        "error":            None,
    }

    def log(msg: str):
        result["pipeline_log"].append(msg)
        if status_cb:
            status_cb(msg)

    # ── STEP 1: Retriever Agent ───────────────────────────────────────────────
    log(" Retriever Agent: Searching document index…")
    try:
        chunks = retrieve(query, top_k=top_k)
        if not chunks:
            result["error"] = "No relevant content found in the indexed documents."
            return result

        context = format_context(chunks)
        result["retrieved_chunks"] = chunks
        result["sources"] = list({c["source"] for c in chunks})
        log(f"   ✓ Found {len(chunks)} relevant chunks from: {', '.join(result['sources'])}")
    except Exception as e:
        result["error"] = f"Retriever failed: {str(e)}"
        return result

    # ── STEP 2: Explainer Agent ───────────────────────────────────────────────
    log(" Explainer Agent: Generating technical explanation…")
    try:
        technical = explainer_agent(query, context)
        result["technical_answer"] = technical
        log("   ✓ Technical explanation generated.")
    except Exception as e:
        result["error"] = f"Explainer Agent failed: {str(e)}"
        return result

    # ── STEP 3: Simplifier Agent ──────────────────────────────────────────────
    log(" Simplifier Agent: Creating student-friendly version…")
    try:
        simple = simplifier_agent(technical, query)
        result["simple_answer"] = simple
        log("   ✓ Simple explanation ready.")
    except Exception as e:
        log(f"   ⚠ Simplifier failed ({e}), using technical answer.")
        result["simple_answer"] = technical   # fallback

    # ── STEP 4: Critic Agent ──────────────────────────────────────────────────
    log(" Critic Agent: Verifying factual correctness…")
    try:
        final = critic_agent(technical, context, query)
        result["final_answer"] = final
        log("   ✓ Critic review complete.")
    except Exception as e:
        log(f"   ⚠ Critic failed ({e}), returning explainer output.")
        result["final_answer"] = technical   # fallback

    log(" Pipeline complete.")
    return result
