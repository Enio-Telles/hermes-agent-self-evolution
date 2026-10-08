# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► Saved artifact ──► Reviewed PR (manual)
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

## Candidate testing and publication (Phase 1)

Skill evolution runs the Hermes test suite by default against a **temporary copy**
of the target repository with the evolved `SKILL.md` substituted at its original
relative path. The installed skill and working tree stay unchanged. Use
`--no-run-tests` only for exploratory runs; it skips this gate.

The command writes `evolved_skill.md`, `baseline_skill.md`, and `metrics.json`
under `output/`. **It does not automatically deploy the candidate or create
a pull request.** A maintainer must inspect the diff, check behavior-specific
tests, and submit a separate reviewed change. A passing general pytest suite
does not by itself establish that the evolved skill's behavior is correct.

## Governed Fitness v2 and skill learning log

The default skill optimization metric uses `LLMJudge` to produce a structured
correctness/procedure/conciseness score plus **textual feedback** for GEPA.
Each judgment receives the instructions actually used by that prediction,
including the evolved candidate's instructions.

The skill optimizer refuses model data egress by default. Set
`HERMES_EVOLVE_API_BASE` to a loopback URL for a local provider, or supply
`--allow-remote-data` **only after** approving the remote model destination
for the skill text, evaluation tasks, and model outputs. This preflight is
not a sandbox and does not govern unrelated model callers.
This sends the approved evaluation tasks, outputs and original skill instructions
to the configured evaluation model; review data sensitivity and provider routing
before running with real sessions. Avoid sensitive/secret-containing datasets.

`--metric-mode heuristic` remains available for inexpensive experiments,
but **cannot** make a candidate eligible for review. The default
`--min-improvement 0.02` requires a positive absolute holdout score gain.
Candidate constraints and test-suite completion are mandatory for review
eligibility. `--no-run-tests` is permitted for exploration but will mark the
candidate as rejected for review purposes. No mode installs or publishes a skill.

A metadata-only SQLite record is written to `output/skill_learning.sqlite3`.
It contains hashes of baseline/candidate texts, dataset fingerprint, model
identifiers, score summaries, test status and review decision. It never writes
raw prompts, skills, responses or session traces to that database. Run artifacts
may still contain candidate text and datasets on disk; treat those directories
as sensitive when used with real session data.

The log captures completed comparisons and early candidate/test/optimizer
rejections, but it is not yet a continuous feedback service. Human review is
still required to actually publish the candidate.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
