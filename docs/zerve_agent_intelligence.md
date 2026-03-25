# Zerve Agent Intelligence Briefing

Understanding the Zerve agent's behavior and cost model was critical to
working effectively within the platform. This document captures what we
learned through hands-on iteration.

## Agent Behavior Model

### Context Window Mechanics

The Zerve agent operates with full canvas awareness. Every prompt -- regardless
of length -- includes the entire canvas state as context. For our 9-block
pipeline (~900 lines of code), this means substantial input token volume
before the user's prompt is even considered.

We confirmed this empirically: a 10-word operational prompt ("rerun upstream
blocks, remove pip install") cost $3, while a detailed 45-line code-change
prompt cost $6. The difference is marginal relative to the fixed context cost.

### Agent Versioning Behavior

The agent maintains an internal model of each block's "canonical" state based
on its own prior edits. When we pasted reconciled code into a block and
prompted the agent to modify it, the agent reverted to its own prior version
and applied changes there instead.

This was a significant discovery. The agent does not simply read the current
block contents -- it has memory of what it previously wrote, and defaults to
that version unless explicitly overridden. We solved this by including
block-specific anchoring details in prompts: algorithm name, row counts,
expected metric values. This forced the agent to recognize it was working
with different code than its internal state.

### Upstream Rehydration Effect

Asking the agent to "run upstream blocks" does more than execute code -- it
restores the agent's canonical versions of those blocks. In our case, we had
pasted reconciled code into block 08 (XGBoost, 300 rows, 13 features), but
when we asked the agent to run the upstream pipeline, blocks 01-07 reverted
to the agent's original versions. This produced the agent's original dataset
(2,960 rows, 28 features) instead of ours (300 rows, 19 columns), and block
08's results reflected the wrong data even if the code was correct.

The fix: paste ALL blocks via the UI before prompting, and never ask the
agent to run blocks selectively. The agent should receive a single
execute-only prompt after all code is in place.

### Execution Cost Structure

The agent appears to use a multi-step reasoning chain for every interaction:
read canvas state, plan changes, generate code, execute blocks, ingest
output, compose response. Each step likely involves a separate LLM call.
Even prompts that only request execution (not code changes) trigger the full
chain, including output analysis.

This means there is a cost floor of approximately $3 per interaction,
regardless of what you ask.

## Workflow We Developed

These observations led us to a specific workflow pattern:

**Local-first development.** All code is written and validated against ground
truth scripts locally before touching the canvas. This eliminates exploratory
prompting inside Zerve.

**Paste-then-prompt.** Reconciled code is pasted into the canvas via the UI
(free), followed by a single prompt that requests all changes at once. The
prompt includes anchoring context so the agent works from the pasted code,
not its internal state.

**UI for operations.** Running blocks, installing packages, managing the
environment, and reordering the DAG are all done through the Zerve UI.
These are zero-cost operations that the agent would charge $3+ to perform.

**Local review.** Agent output is copied back to the local environment for
validation against acceptance criteria. We never ask the agent to
self-evaluate its own changes.

**Batch aggressively.** Since prompt count drives cost more than prompt
length, we combine multiple changes into single prompts wherever possible.

## Environment Observations

- The default Python environment does not include `xgboost`. The ML Starter
  environment does. Selecting the right environment before prompting avoids
  wasted agent interactions on dependency installation.
- The agent's fallback for missing packages is `subprocess.run(['pip',
  'install', ...])` injected at the top of the block. This works but adds
  runtime overhead on every execution and consumes agent tokens to generate.
  Pre-installing via the environment UI is strictly better.
- Blocks share a variable namespace within a canvas session. Upstream blocks
  must run before downstream blocks that reference their outputs (e.g.,
  `df_features`). The agent understands DAG order but may not automatically
  re-run upstream dependencies.

## Lessons Learned

1. **The agent is a code generator, not an IDE.** Treat it as a service that
   transforms specifications into code. Everything else -- execution, debugging,
   environment setup -- is cheaper and faster through the UI.

2. **Anchoring prevents version drift.** Always include concrete details
   (algorithm, row counts, expected values) when prompting changes to blocks
   that contain reconciled or externally-written code.

3. **Cost scales with canvas size, not prompt size.** As the canvas grows,
   every interaction becomes more expensive. This favors building blocks
   correctly the first time over iterative agent-assisted refinement.

4. **Front-load complexity into fewer prompts.** A detailed first prompt that
   gets it right costs less than a vague first prompt plus two correction
   prompts.
