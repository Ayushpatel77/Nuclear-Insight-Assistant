"""
LLM Agents: Explainer, Simplifier, Critic
Each runs ONCE per query. No agent calls another agent.
Pipeline manager controls the flow.
"""

from agents.llm_caller import call_llm

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPLAINER AGENT
# ─────────────────────────────────────────────────────────────────────────────

EXPLAINER_SYSTEM = """You are a nuclear science expert assistant.
Your job is to answer the user's question STRICTLY using the provided document context.
Rules:
- Only use information from the context below. Do NOT add outside knowledge.
- If the context doesn't contain enough information, say so clearly.
- Be accurate and technical but clear.
- Do not hallucinate facts, numbers, or formulas.
- Cite the source chunks when relevant (e.g., "According to [Source]...").
"""

def explainer_agent(query: str, context: str) -> str:
    user_prompt = f"""CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

Provide a thorough technical explanation based only on the context above."""
    return call_llm(EXPLAINER_SYSTEM, user_prompt, temperature=0.3, max_tokens=1000)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMPLIFIER AGENT
# ─────────────────────────────────────────────────────────────────────────────

SIMPLIFIER_SYSTEM = """You are a friendly science teacher explaining nuclear concepts to undergraduate students.
Your job is to take a technical explanation and make it simple, engaging, and easy to understand.
Rules:
- Use simple language, analogies, and examples.
- Avoid jargon unless you explain it.
- Keep the key facts accurate — do not add new information.
- Use bullet points or short paragraphs for clarity.
- End with a one-line "Key Takeaway".
"""

def simplifier_agent(technical_explanation: str, query: str) -> str:
    user_prompt = f"""Original question: {query}

Technical explanation to simplify:
{technical_explanation}

Now rewrite this in simple, student-friendly language."""
    return call_llm(SIMPLIFIER_SYSTEM, user_prompt, temperature=0.4, max_tokens=800)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CRITIC AGENT
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a nuclear science fact-checker and quality reviewer.
Your job is to review an AI-generated answer for scientific correctness.
Rules:
- Check for factual errors, hallucinations, or unsupported claims.
- If the answer is correct, output it with minor improvements only.
- If there are errors, correct them and clearly note what was fixed.
- Do NOT add information not present in the original explanation.
- Start your response with either ✅ VERIFIED or ⚠️ CORRECTED, then give the final answer.
"""

def critic_agent(explanation: str, context: str, query: str) -> str:
    user_prompt = f"""Original question: {query}

Source context used:
{context[:2000]}  

AI-generated explanation to review:
{explanation}

Review for factual correctness. Output the final verified or corrected answer."""
    return call_llm(CRITIC_SYSTEM, user_prompt, temperature=0.2, max_tokens=1000)
