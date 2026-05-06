"""
LLM Caller — OpenRouter API wrapper.
All three LLM agents (Explainer, Simplifier, Critic) use this.
"""

import os
import json
import requests
from typing import Optional


OPENROUTER_API_KEY = os.environ.get("", "Your Open Router API Key")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL      = "nvidia/nemotron-3-nano-30b-a3b:free"   

def call_llm(
    system_prompt: str,
    user_prompt:   str,
    model:         str = DEFAULT_MODEL,
    temperature:   float = 0.3,
    max_tokens:    int  = 1024,
) -> str:
    """
    Call OpenRouter LLM and return the response text.
    Raises RuntimeError on failure.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://nuclear-insight.local",
        "X-Title":       "Nuclear Insight Assistant",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OpenRouter API error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {str(e)}")
