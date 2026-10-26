# Contributing to agent-runtime

Thank you for helping improve Kionga. Read [ARCHITECTURE.md](ARCHITECTURE.md) for
how the packages fit together before you make a larger change.

## Development setup

Prerequisites:

- Python 3.11, the version of the image and of CI
- Docker with BuildKit, to build the image
- Go, any recent stable release, only to run the gitleaks secret scan locally

```bash
git clone git@github.com:kiongahq/agent-runtime.git
cd agent-runtime
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs `mlaiops-sdk[runtime,dev]` from the SDK release this
runtime is tested against. To work against a sibling SDK checkout instead:

```bash
pip install -e "../sdk-python[runtime,dev]"
```

## Run the checks

CI calls the shared Python workflow
([`kiongahq/.github/.github/workflows/python.yml`](https://github.com/kiongahq/.github/blob/main/.github/workflows/python.yml))
with `pip install -r requirements-dev.txt`. Run the same steps locally:

```bash
ruff check .
python -m pytest -q
```

Then the security gates, with the versions CI pins:

```bash
pip install --upgrade pip setuptools "pip-audit==2.10.1"
pip-audit --skip-editable                                    # known vulnerabilities in installed packages
pip install "bandit==1.9.4"
bandit -q -r . -x ./tests,./.venv,./node_modules -ll         # medium and higher static findings
go run github.com/zricethezav/gitleaks/v8@v8.30.1 git --no-banner --redact .   # secrets in the full history
```

The gate tests use the deterministic `mock` LLM and fake HTTP endpoints, so they
need no network, no database and no provider key.

## Run the agent evals

Evals measure answer quality with a real LLM, so they cost money and never run in
CI. Run them before you ship a change to an agent or a prompt:

```bash
MLAIOPS_RUN_EVALS=1 MLAIOPS_LLM_BACKEND=openai OPENAI_API_KEY=... \
  python -m pytest -q agent_runtime/evals/eval_customer_support.py agent_runtime/evals/eval_native_graph.py
```

The customer support eval scores each case in `agent_runtime/evals/golden.jsonl`
with an LLM judge. It passes when the mean score is at least 0.7 and no case scores
below 0.4. Without `MLAIOPS_RUN_EVALS=1` the eval is skipped. Never run evals from
an untrusted pull request with provider keys exposed.

## Build the image

```bash
# SDK from its tagged release, as CI builds it
docker build -t ghcr.io/kiongahq/agent-runtime:dev .

# SDK from a sibling checkout, for a change that spans both repositories
docker build --build-arg SDK_SOURCE=local --build-context sdk=../sdk-python \
  -t ghcr.io/kiongahq/agent-runtime:dev .
```

`SDK_VERSION` in the `Dockerfile` and the pin in `requirements-dev.txt` name the
same SDK release. Bump both together.

On pushes to `main` and on `v*` tags, CI builds the image with the shared image
workflow and publishes `ghcr.io/kiongahq/agent-runtime` tagged `sha-<short>`,
`latest` (from `main`) and the semantic version (from tags). Pull requests build the
image without publishing it.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/), as the history
does: `type(scope): summary` in the imperative mood, for example
`fix(features): wait for materializer dependencies` or
`chore(security): record bandit review of the container listener`. Common types are
`feat`, `fix`, `docs`, `test`, `ci` and `chore`.

## Pull requests

- Keep one logical change per pull request, with tests for new behavior.
- A change to an agent's behavior includes an eval run, with the scores in the pull
  request description.
- CI must pass. Do not skip hooks or checks.

## Reviewed scanner findings

When bandit flags code that you have reviewed and found safe, annotate the line.
Put the reason in a comment on the line before it, and end the flagged line with
`# nosec <rule>` and nothing after it (bandit reads any text after `nosec` as rule
IDs):

```python
# All interfaces inside the container; MLAIOPS_RUNTIME_HOST overrides.
host=os.environ.get("MLAIOPS_RUNTIME_HOST", "0.0.0.0"),  # nosec B104
```

If a test needs a fake credential that gitleaks reports, add the finding's
fingerprint (`commit:file:rule:line`, printed by gitleaks) to a `.gitleaksignore`
file at the repository root, with a comment that says why. A fingerprint pins one
exact finding, so any new secret still fails the scan.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENCE).
