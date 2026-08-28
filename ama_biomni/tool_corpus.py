"""Load Biomni's real tool corpus (module -> list of tool schemas).

Biomni groups its ~220 biomedical tools into per-domain modules (genetics,
genomics, cancer_biology, ...) and exposes them as ``module2api`` via
``biomni.utils.read_module2api()``. Each tool schema looks like::

    {"name": "liftover_coordinates", "description": "...",
     "required_parameters": [{"name": "chromosome", "type": "str", "description": "...", "default": None}, ...],
     "optional_parameters": [...]}

This is the direct analog of AMA's ``data/all_normal_tools.jsonl`` (its
pyopenagi tool list, grouped by "Corresponding Agent"). Here the grouping key
is the Biomni module name instead of an agent name.

Reading ``read_module2api()`` only imports plain-data ``tool_description``
modules -- it does not touch Biomni's ~11GB datalake or require API keys, so
this works even in a lightweight ``pip install biomni`` environment. If
``biomni`` isn't installed at all, we fall back to a static snapshot of the
same corpus bundled at ``data/biomni_tool_corpus_snapshot.json`` (captured
from the upstream repository) so the rest of the harness still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "biomni_tool_corpus_snapshot.json"


def load_module2api() -> dict[str, list[dict]]:
    """Return ``{module_name: [tool_schema, ...]}`` for every Biomni tool module."""
    try:
        from biomni.utils import read_module2api

        return read_module2api()
    except Exception:
        with _SNAPSHOT_PATH.open() as f:
            return json.load(f)


def flatten_tools(module2api: dict[str, list[dict]]) -> list[dict]:
    """Flatten ``module2api`` into a single list, each tool tagged with its module."""
    flat = []
    for module_name, tools in module2api.items():
        for tool in tools:
            flat.append({**tool, "module": module_name})
    return flat


def normal_tool_params(module2api: dict[str, list[dict]]) -> list[dict]:
    """Collect ``{name, description}`` for every real tool *parameter*.

    Used as few-shot "reference style" material when generating adversarial
    parameter names/descriptions in stage 2 -- the same role AMA's
    ``data/normal_para_desc.jsonl`` plays.
    """
    params = []
    for tools in module2api.values():
        for tool in tools:
            for param in tool.get("required_parameters", []) + tool.get("optional_parameters", []):
                params.append({"name": param["name"], "description": param.get("description", "")})
    return params


if __name__ == "__main__":
    m2a = load_module2api()
    total = sum(len(v) for v in m2a.values())
    print(f"Loaded {len(m2a)} modules, {total} tools.")
