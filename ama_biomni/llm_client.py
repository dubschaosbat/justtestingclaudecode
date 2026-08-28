"""Chat helpers shared by every pipeline stage.

Ported from AMA's ``llm_utils/utils.py``. AMA talked to models through a raw
OpenAI-compatible client (``GPT_API_BASE``/``GPT_API_KEY`` or an Xinference
endpoint). Here we prefer Biomni's own ``biomni.llm.get_llm`` factory when the
``biomni`` package is installed, so the harness authenticates against whatever
model source a real Biomni deployment already uses (Anthropic, OpenAI, Gemini,
Bedrock, Groq, Ollama, or a custom endpoint). When ``biomni`` isn't installed we
fall back to a plain OpenAI-compatible client, matching AMA's original behavior.
"""

from __future__ import annotations

import ast
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def eval_from_json(response: str) -> dict | None:
    """Best-effort extraction of a JSON object embedded in a chat completion.

    Ported near-verbatim from AMA's ``eval_from_json`` -- models routinely wrap
    their JSON answer in ```json fences, plain fences, or nothing at all.
    """
    if response is None:
        return None
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.lstrip("\n")
    try:
        match = re.search(r"```json(.*?)```", response, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(1).strip())

        match = re.search(r"```(.*?)```", response, re.DOTALL)
        if match:
            content = match.group(1).strip().removeprefix("json").strip()
            return ast.literal_eval(content)

        return ast.literal_eval(response.strip())
    except Exception:
        return None


@lru_cache(maxsize=None)
def _get_langchain_llm(model: str):
    """Build a LangChain chat model for ``model``, preferring Biomni's own factory."""
    try:
        from biomni.llm import get_llm

        return get_llm(model=model)
    except Exception:
        pass

    from langchain_openai import ChatOpenAI

    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("GPT_API_BASE") or os.getenv("XINFERENCE_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY") or os.getenv("XINFERENCE_API_KEY") or "EMPTY"
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, timeout=240, max_retries=2)


def chat(prompt: str, model: str = "gpt-4o-mini", system: str | None = None) -> str:
    """Synchronous single-turn chat, returning the model's text content."""
    llm = _get_langchain_llm(model)
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return re.sub(r".*?</think>", "", content, flags=re.DOTALL)


async def achat(prompt: str, model: str = "gpt-4o-mini", system: str | None = None) -> str:
    """Async single-turn chat, returning the model's text content."""
    llm = _get_langchain_llm(model)
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return re.sub(r".*?</think>", "", content, flags=re.DOTALL)


def chat_and_get_json(prompt: str, model: str = "gpt-4o-mini", max_tries: int = 3) -> dict | None:
    for _ in range(max_tries):
        result = eval_from_json(chat(prompt, model=model))
        if result is not None:
            return result
    return None


async def achat_and_get_json(prompt: str, model: str = "gpt-4o-mini", max_tries: int = 3) -> dict | None:
    for _ in range(max_tries):
        result = eval_from_json(await achat(prompt, model=model))
        if result is not None:
            return result
    return None
