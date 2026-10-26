# Agent evals (paid lane)

Quality-measuring evals for platform agents. These call a real LLM, so they
never run in the commit gate; run them before ship and nightly.

## customer-support

- **Golden set:** `golden.jsonl` — question, customer entity, expected facts.
- **Method:** run the agent graph per case, then LLM-as-judge scores answer
  faithfulness against the expected facts (0.0–1.0).
- **Threshold:** mean >= 0.7 and every case >= 0.4. Below threshold fails the run.

## native-graph

- **Golden set:** `native_graph_golden.jsonl` — template, input and expected
  fields (structured extraction) or expected branch and keywords (customer support).
- **Method:** run the starter template through the native interpreter with the real
  LLM, then score deterministically: matching JSON fields, or correct route plus
  keyword coverage (0.0–1.0).
- **Threshold:** mean >= 0.7 and every case >= 0.4.

## judge

- **Golden set:** `judge_golden.jsonl`: one good and one bad reply per case for
  each rubric criterion (groundedness, task completion, relevance, safety),
  including a bad reply that tries to instruct the judge.
- **Method:** score both replies with the designer runner's rubric judge
  (`agent_runtime/native_graph/judge.py`). A pair passes when good >= 0.6,
  bad <= 0.4 and the gap is >= 0.3.
- **Threshold:** >= 0.85 of pairs pass, and the prompt-injection pair must pass.
  Run it whenever the judge model, prompt or rubric version changes.

```bash
MLAIOPS_RUN_EVALS=1 MLAIOPS_LLM_BACKEND=openai OPENAI_API_KEY=... \
  python -m pytest python/agent_runtime/evals -q
```

The harness prints per-case scores. Production quality gates should additionally
record the dataset version, agent image, graph module, prompt/model version, Git
commit, and aggregate scores in MLflow or Langfuse so promotion decisions are
reproducible. Never run paid evals from an untrusted pull request with provider
secrets exposed.
