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
import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _parse_object(text: str) -> dict | None:
    """Parse ``text`` as a JSON object, trying real JSON first.

    ``json.loads`` handles standard JSON escaping (``\\n`` inside a string,
    ``true``/``false``/``null``) that ``ast.literal_eval`` -- a *Python*
    literal parser, not a JSON parser -- can choke on even for well-formed
    JSON. AMA's original ``eval_from_json`` only tried ``ast.literal_eval``;
    that was enough for the shorter completions it saw from GPT-4o-mini/Qwen,
    but Claude's longer, more carefully-escaped JSON strings trip it up.
    ``ast.literal_eval`` stays as a fallback for models that emit
    Python-literal-styled output (e.g. single-quoted strings).
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def eval_from_json(response: str) -> dict | None:
    """Best-effort extraction of a JSON object embedded in a chat completion.

    Ported from AMA's ``eval_from_json`` -- models routinely wrap their JSON
    answer in ```json fences, plain fences, or nothing at all.
    """
    if response is None:
        return None
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.lstrip("\n")

    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        result = _parse_object(match.group(1).strip())
        if result is not None:
            return result

    match = re.search(r"```(.*?)```", response, re.DOTALL)
    if match:
        content = match.group(1).strip().removeprefix("json").strip()
        result = _parse_object(content)
        if result is not None:
            return result

    return _parse_object(response.strip())


@lru_cache(maxsize=None)
def _get_langchain_llm(model: str):
    """Build a LangChain chat model for ``model``, preferring Biomni's own factory.

    Without ``biomni`` installed, fall back to source-detecting from the model
    name prefix the same way ``biomni.llm.get_llm`` does -- a ``claude-*``
    model must go through ``ChatAnthropic``/``ANTHROPIC_API_KEY``, not
    ``ChatOpenAI``, or the request silently gets routed to OpenAI's real
    endpoint with an empty key.
    """
    try:
        from biomni.llm import get_llm

        return get_llm(model=model)
    except Exception:
        pass

    if model.startswith("claude-"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                f"Model '{model}' looks like a Claude model but `biomni` isn't installed and "
                "`langchain-anthropic` isn't either. Run: pip install langchain-anthropic"
            ) from e
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(f"Model '{model}' needs ANTHROPIC_API_KEY set in the environment.")
        return ChatAnthropic(model=model, api_key=api_key, timeout=240, max_retries=2)

    if model.startswith("gemini-"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                f"Model '{model}' looks like a Gemini model but `biomni` isn't installed and "
                "`langchain-google-genai` isn't either. Run: pip install langchain-google-genai"
            ) from e
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Model '{model}' needs GOOGLE_API_KEY (or GEMINI_API_KEY) set in the environment.")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, timeout=240, max_retries=2)

    from langchain_openai import ChatOpenAI

    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("GPT_API_BASE") or os.getenv("XINFERENCE_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY") or os.getenv("XINFERENCE_API_KEY")
    if not api_key and base_url is None:
        raise RuntimeError(
            f"Model '{model}' doesn't match a known Claude/Gemini prefix, so it's being routed to "
            "OpenAI's ChatOpenAI client, but no OPENAI_API_KEY (or GPT_API_KEY/XINFERENCE_API_KEY + "
            "matching *_API_BASE for a custom endpoint) is set."
        )
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key or "EMPTY", timeout=240, max_retries=2)


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


class JsonParseError(RuntimeError):
    """Raised when a model's response never yielded parseable JSON after retries.

    Carries the last raw response so callers/tracebacks show *why* it failed
    (a refusal, a format the regex-based extractor doesn't recognize, etc.)
    instead of failing later and cryptically wherever the caller does
    ``result["field"]`` on a silent ``None``.
    """

    def __init__(self, last_response: str, max_tries: int):
        self.last_response = last_response
        preview = (last_response or "")[:4000]
        super().__init__(f"Model returned no parseable JSON after {max_tries} tries. Last response:\n{preview}")


def chat_and_get_json(prompt: str, model: str = "gpt-4o-mini", max_tries: int = 3) -> dict:
    last_response = ""
    for _ in range(max_tries):
        last_response = chat(prompt, model=model)
        result = eval_from_json(last_response)
        if result is not None:
            return result
    raise JsonParseError(last_response, max_tries)


async def achat_and_get_json(prompt: str, model: str = "gpt-4o-mini", max_tries: int = 3) -> dict:
    last_response = ""
    for _ in range(max_tries):
        last_response = await achat(prompt, model=model)
        result = eval_from_json(last_response)
        if result is not None:
            return result
    raise JsonParseError(last_response, max_tries)
