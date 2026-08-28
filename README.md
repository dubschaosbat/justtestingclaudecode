# AMA-for-Biomni

An **Attractive Metadata Attack (AMA)** harness retargeted at [Biomni](https://github.com/snap-stanford/biomni),
Stanford SNAP's general-purpose biomedical research agent. This reproduces the methodology from:

> Mo, Hu, Long, Li. *Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools.*
> NeurIPS 2025, Poster. https://neurips.cc/virtual/2025/loc/san-diego/poster/116046

The original [AMA implementation](https://github.com/SEAIC-M/AMA) targets the `aios`/`pyopenagi`
(OpenAGI) agent framework. Biomni is architecturally unrelated to that framework, so nothing there
ports 1:1 -- this repo re-derives AMA's four-stage pipeline against Biomni's actual tool registry,
retrieval mechanism, and code-execution loop.

**Intended use: authorized security research and robustness evaluation.** This crafts tool metadata
designed to get selected and invoked over legitimate tools, and to elicit sensitive parameter values
during invocation. Use it against your own Biomni deployment, or one you're explicitly authorized to
test, to evaluate and harden its tool-selection pipeline -- the same posture as the original AMA paper.

## How this maps onto AMA

| AMA (pyopenagi/aios) | This repo (Biomni) |
|---|---|
| `construct_privacy_info/construct_memory.py` (ai4privacy PII dataset) | `ama_biomni/secrets_dataset.py` -- synthetic biomedical-context secrets (patient/sample IDs, genomic variants, lab API keys, ...), generated locally, no network dependency |
| `construct_privacy_info/generate_privacy_parameter.py` | `ama_biomni/generate_secret_parameters.py` -- same idea, reference style drawn from Biomni's own ~220 real tool parameters |
| pyopenagi tool list, grouped by "Corresponding Agent" (`data/all_normal_tools.jsonl`) | Biomni's real `module2api` tool corpus, grouped by "Corresponding Module" (`ama_biomni/tool_corpus.py`) |
| `generate_tool_metadata.py` (target/untarget generation + reconstruct loop) | `ama_biomni/tool_metadata_gen.py` -- same generate/evaluate/reconstruct loop, success measured by mirroring Biomni's *actual* `ToolRetriever.prompt_based_retrieval` index-selection prompt |
| `pyopenagi/tools/simulated_tool.py::AttackerTool` | `ama_biomni/attacker_tool.py` -- builds a live callable + Biomni-shaped schema (`required_parameters` list, not OpenAI JSON-schema) |
| `ReactAgentAttack` / `llm_utils/agent_worker.py` | `ama_biomni/biomni_adapter.py` -- `RealBiomniAdapter` (wraps a live `biomni.agent.a1.A1`) and `MockBiomniAdapter` (offline stand-in with the same interface, for testing without a full Biomni install) |
| `main_attacker.py` (orchestration, Excel export) | `ama_biomni/main_attacker.py` -- same three metrics, same `.xlsx` export |

Two honest scoping differences from AMA, called out because they'd matter for a writeup:

1. **Secret dataset**: AMA downloads a public PII dataset. Biomni's real context surface (patient IDs,
   genomic variants, lab credentials) has no equivalent off-the-shelf dataset, so `secrets_dataset.py`
   generates synthetic records locally instead.
2. **Original-task-success metric**: AMA can check this exactly because every pyopenagi toy tool reports
   a fixed `expected_achivement` string. Biomni's ~220 real tools have no such canned success string, so
   `metrics.original_task_progressed` is a documented heuristic, not an exact check.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Stages 1-3 (secret dataset, secret parameters, adversarial tool generation) need only an LLM -- no
Biomni install required, since `tool_corpus.py` falls back to a bundled snapshot of Biomni's real tool
corpus (`data/biomni_tool_corpus_snapshot.json`) when `biomni` isn't installed. Configure model access
via `.env` (same variables Biomni itself reads, e.g. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), or
`OPENAI_API_BASE`/`OPENAI_API_KEY` for a custom OpenAI-compatible endpoint.

To run stage 4 against a **real** Biomni agent (`--mode real`), additionally:

```bash
pip install -e ".[biomni]"
# follow Biomni's own setup for its datalake / conda env, per its README
```

Without that, stage 4 defaults to `--mode mock`, which runs the full attack chain (retrieval, code
generation, tool invocation, parameter capture) against `MockBiomniAdapter` -- useful for iterating on
attack design before spending real API calls or needing the ~11GB datalake.

> **Security note:** this harness executes LLM-generated code (`MockBiomniAdapter`, and Biomni itself in
> the real adapter). Run it only inside an isolated container or VM, exactly as upstream Biomni
> recommends for its own code executor.

## Quickstart

```bash
# 1. Build the synthetic secret dataset
python -m ama_biomni.secrets_dataset

# 2. Generate persuasive secret-eliciting tool parameters
python -m ama_biomni.generate_secret_parameters --llm_name gpt-4o-mini

# 3. Generate + refine adversarial tool metadata (target attack, per Biomni module)
python -m ama_biomni.tool_metadata_gen --llm_name gpt-4o-mini --attack_type target -t 0.9 --lambda_weight 0.5

# 4. Run the generated attacker tools live (mock adapter by default)
python -m ama_biomni.main_attacker --llm_name gpt-4o-mini --attack_type target \
    --run_tools_num 2 --steal_params_num 5 --task_num 3 --mode mock
```

Results:
- `generate_tools/<llm_name>/<attack_type>/result.xlsx` -- generated tools + their selection (Success) rate.
- `logs/attack_results.xlsx` -- per-run Attack Success, Secret Steal Rate, and Original Task Progressed.

Untargeted variant:

```bash
python -m ama_biomni.tool_metadata_gen --llm_name gpt-4o-mini --attack_type untarget -t 0.8
python -m ama_biomni.main_attacker --llm_name gpt-4o-mini --attack_type untarget --mode mock
```

## Integrating with a DoomArena-based Biomni sandbox

If you already have [DoomArena](https://github.com/ServiceNow/DoomArena) wired up against a real Biomni
install -- e.g. via a hand-written `AttackGateway` that monkeypatches
`ToolRetriever.prompt_based_retrieval` to inject text into whatever it returns (indirect prompt
injection / observation poisoning) -- that's a *different* attack vector from this repo's core
contribution, not a duplicate of it: it mutates retrieval's output after the fact, where AMA's actual
mechanism is a new tool competing for selection on its own crafted metadata. Both patch the same seam
and can be run side by side against the same model for comparison.

`ama_biomni/doomarena_gateway.py` bridges the two: `ToolRegistrationAttackGateway` registers a live
AMA-style malicious tool into a real `A1` agent (via the same agent-state splice
`ama_biomni/attacker_tool.py` uses elsewhere) and reports, through DoomArena's own `SuccessFilter`
pattern, whether it was invoked (`AttackerToolInvoked`) and whether a planted secret leaked through its
call parameters (`SecretExfiltrated`). It requires `doomarena` installed in your sandbox; nothing else in
this repo depends on it.

`run_ama_attack.py` is a ready-to-drop-in script matching the construction pattern of a
`run_clean_control.py`/`run_canary_attack.py` pair (`A1(path="./data", llm=model, timeout_seconds=300,
expected_data_lake_files=[])` -- confirmed to avoid pulling Biomni's full datalake). It prefers a real
attacker tool from stage 3's `generate_tools/<llm>/target/result.xlsx` if you've generated one, falling
back to a built-in example so it runs with zero setup:

```bash
pip install -e .  # in your sandbox, so `ama_biomni` is importable next to `doomarena` and `biomni`
BIOMNI_TEST_MODEL=claude-sonnet-4-5-20250929 python run_ama_attack.py --module biomni.tool.genetics
```

> **Note:** an attacker tool's `required_parameters` use persuasive names generated in stage 2 (e.g.
> `patient_identifier`), not the canonical secret key (`PATIENT_ID`) verbatim -- that's the point of the
> attack. `metrics.secret_steal_rate` needs the `{param_name: secret_key}` map from stage 2's output (or
> `SECRET_PARAMS` in `run_ama_attack.py`) to translate a captured call back to the real secret before
> comparing; passing raw call params without it will silently read as zero leakage. This mirrors AMA's
> own `PARAMS_TO_PRIVACY_KEY` translation in its `main_attacker.py`.

## Tests

```bash
pytest
```

Every test is offline and deterministic -- no API keys or Biomni install required. They verify the
plumbing (schema shapes, agent-state splicing, metric math, the full mock attack chain) rather than
real attack effectiveness, which needs a live model.

## Acknowledgments

Methodology and prompt structure adapted from the official
[AMA implementation](https://github.com/SEAIC-M/AMA), which in turn credits the
[Agent Security Bench (ASB)](https://github.com/agiresearch/ASB) project. Retargeted at
[Biomni](https://github.com/snap-stanford/biomni) (Stanford SNAP lab).

```bibtex
@inproceedings{mo2025ama,
  title={Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools},
  author={Mo, Kanghua and Hu, Li and Long, Yucheng and Li, Zhihao},
  booktitle={Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025), Poster},
  year={2025},
  url={https://neurips.cc/virtual/2025/loc/san-diego/poster/116046}
}
```
