# Agent evals (paid lane)

Quality-measuring evals for platform agents. These call a real LLM, so they
never run in the commit gate; run them before ship and nightly.

## customer-support

- **Golden set:** `golden.jsonl` — question, customer entity, expected facts.
- **Method:** run the agent graph per case, then LLM-as-judge scores answer
  faithfulness against the expected facts (0.0–1.0).
- **Threshold:** mean >= 0.7 and every case >= 0.4. Below threshold fails the run.

```bash
MLAIOPS_RUN_EVALS=1 MLAIOPS_LLM_BACKEND=openai OPENAI_API_KEY=... \
  python -m pytest python/agent_runtime/evals -q
```

The harness prints per-case scores. Production quality gates should additionally
record the dataset version, agent image, graph module, prompt/model version, Git
commit, and aggregate scores in MLflow or Langfuse so promotion decisions are
reproducible. Never run paid evals from an untrusted pull request with provider
secrets exposed.
