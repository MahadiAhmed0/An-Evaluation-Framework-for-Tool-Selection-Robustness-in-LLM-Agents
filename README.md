# Tool-Selection Robustness Evaluation Harness

A Python research harness for **benchmarking tool-selection robustness in LLM
agents** and for **evaluating detection methods**. It models the standard
"retrieve -> select" pipeline used by LLM agent frameworks (tool documents, an
embedding retriever, a two-step selection prompt) and measures how reliably an
agent selects the intended tool â€” including under modified tool registries.

> **Scope note:** this project is defensive research tooling. It exists to
> *measure* selection/retrieval behavior and detection quality. It does not
> generate or optimize attacks.

## What it measures

| Metric | Meaning |
| --- | --- |
| `accuracy` | Fraction of baseline queries where the expected tool was selected |
| `hit_rate_at_k` | Fraction of queries where the expected tool appeared in the retrieved top-k set |
| `target_selection_rate` | Fraction of passes where a researcher-supplied *test document* (e.g., an injected or variant tool description) was the selected tool |
| `target_retrieval_rate` | Fraction of passes where the test document appeared in the top-k set |
| `selector_status_counts` | Breakdown of selector outcomes (`success`, `invalid_json`, `unknown_tool`, `refused`) |
| FNR / FPR | Detector error rates on a labeled mixed set of benign and test documents |

## Project structure

```
tool_selection_harness/
  core/
    tool_document.py     # ToolDocument dataclass (name + description)
    tool_library.py      # ToolLibrary: add/remove/inject/load/save (copy-on-write)
    retriever.py         # Embedding retriever with pluggable embed_fn + cache
    selector.py          # Two-step selection prompt, robust JSON parsing
    metrics.py           # EvalRecord + benchmark metrics
    runner.py            # BenchmarkRunner: retrieve -> select loop + reporting
    generators.py        # LLM-based synthetic query/tool-document generation
    defenses.py          # Detector protocol, PerplexityDetector, ThresholdClassifier
    detection_metrics.py # FPR/FNR for detector evaluation
  data/
    sample_tools.json    # 15 synthetic tool documents across categories
  examples/
    run_benchmark.py     # End-to-end example (see below)
  tests/                 # pytest suite (112 tests, no network required)
```

## Installation

```
pip install -r requirements.txt
# optional, for the perplexity detector and real LLM APIs:
pip install transformers torch openai anthropic
```

## Web UI

A Streamlit frontend (`app.py`) exposes the whole harness interactively:

```
streamlit run app.py
```

Four tabs:

- **Tool Library** â€” load `data/sample_tools.json` or upload a custom JSON,
  edit documents in an editable table, add/remove tools via a form, and
  generate synthetic tools through `core/generators.py` using the selected
  LLM backend.
- **Run Benchmark** â€” pick the embedding backend (MiniLM or offline
  hashing), similarity metric, and top-k; paste or auto-generate queries;
  optionally inject one hand-written *benign comparison variant* document
  (clearly labeled as metric-testing material, not an attack payload). Runs
  `core/runner.py`'s `BenchmarkRunner` with a progress bar and shows metric
  cards, a baseline-vs-injected bar chart, and raw per-query records.
- **Detection** â€” calibrates `PerplexityDetector` (gpt2, one-time download)
  on the current library, plots the benign score histogram with the variant
  marked, lets you sweep the FPR target (live FNR/FPR and flagged-document
  table), and draws the FNR-vs-FPR tradeoff line.
- **History** â€” saved runs (`benchmark_history/`) can be reloaded and
  compared side by side.

The LLM provider is chosen in the sidebar (Anthropic / OpenAI / mock); all
calls go through the same pluggable `selector.py` / `generators.py`
interfaces, so the UI never hardcodes a provider.

## Alignment with the research paper

This project reproduces the framework of *Prompt Injection Attack to Tool
Selection in LLM Agents* (Shi et al., NDSS 2026 â€” `Related Papers/`) inside
a controlled, defensive evaluation environment:

| Paper element | Module |
| --- | --- |
| Two-step tool selection (retrieval -> selection) | `core/retriever.py`, `core/selector.py` |
| Selection prompt structure (Fig. 2) | `core/selector.py` (`PROMPT_TEMPLATE`) |
| Metrics ACC / ASR / HR@k / AHR@k | `core/metrics.py` (`accuracy`, `target_selection_rate`, `hit_rate_at_k`, `target_retrieval_rate`) |
| Manual attack baselines (naive, escape, context ignore, fake completion, combined) | `core/attacks.py` (`manual_attack_documents`) |
| ToolHijacker gradient-free (R generation + S tree search, Algorithm 1) | `core/attacks.py` (`toolhijacker_gradient_free`) |
| ToolHijacker gradient-based (L1/L2/L3 losses, GCG-style S + HotFlip-style R) | `core/attacks.py` (`GradientSelectionOptimizer`, `GradientRetrievalOptimizer`) |
| PPL / PPL-W / known-answer detection | `core/defenses.py` |
| Dataset-adaptive thresholding + FNR/FPR/AUC (Table X) | `core/defenses.py`, `core/detection_metrics.py` (`evaluate_detector`, `detection_auc`) |
| Query / tool-document generation prompts (Fig. 10/11) | `core/generators.py` |

End-to-end reproduction script:
