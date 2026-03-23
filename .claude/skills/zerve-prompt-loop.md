# zerve-prompt-loop

Iterative prompt-and-ingest loop for working with the Zerve AI agent. Claude crafts
prompts, the user runs them in Zerve, copies the agent's code output back, and Claude
updates local state, reviews, and iterates.

## Trigger

`/zerve-prompt-loop` or "zerve prompt", "agent prompt", "prompt for zerve"

## Workflow

### Phase 1 — Inventory (runs once per session or when blocks change)

1. Read `zerve_blocks/` directory and build a **block manifest**:
   - File name, block number, DAG position, critical-path or exploration
   - SHA256 of current content (to detect when user pastes new code)
   - Status: `baseline` | `prompt-sent` | `agent-updated` | `reviewed`
2. Save manifest to `zerve_blocks/.block_manifest.json`
3. Print manifest table to user

### Phase 2 — Prompt Generation

1. User specifies what improvement(s) to target (or Claude proposes from open beads)
2. Claude crafts a **Zerve agent prompt** — a self-contained message the user can
   paste into the Zerve agent chat. The prompt:
   - References only block names the agent knows (no file paths)
   - Gives the agent enough context to make the right change
   - May include ground-truth code snippets, formulas, or methodology hints when
     the agent needs specific implementation details to get it right. Use judgment:
     guide with methodology and acceptance criteria first, share code when precision
     matters (e.g., purged CV logic, specific feature lists).
   - Includes acceptance criteria so the user can verify output
3. Claude saves the prompt to `zerve_blocks/.prompts/<block>_v<N>.md` for history
   - Versioned: v1 = first attempt, v2+ = follow-ups for same goal
4. Claude updates manifest status to `prompt-sent`

### Phase 3 — Ingest

1. User runs the prompt in Zerve, copies the agent's code output
2. User pastes the code into the conversation (or into the local file directly)
3. Claude:
   - Writes/overwrites the corresponding `zerve_blocks/<block>.py` file
   - Updates manifest SHA and status to `agent-updated`
   - If the agent created a NEW block: adds it to manifest with next available number
   - If the agent DELETED or merged blocks: marks old entries `deleted` in manifest,
     removes local files, updates DAG notes
   - If the agent RENAMED a block: updates manifest entry, renames local file
4. Claude diffs agent output vs expected behavior (from the prompt's acceptance
   criteria) and reports pass/fail per criterion

### Phase 4 — Review & Iterate

1. If the agent output meets acceptance criteria -> status = `reviewed`, done
2. If not -> Claude crafts a follow-up prompt addressing specific gaps, back to Phase 2
   - Follow-up prompt references what the agent got right and what to fix
   - Saved as next version (e.g., `block08_v2.md`)
3. When all targeted blocks are `reviewed`, Claude prints a summary and updates
   any relevant beads

## Block Manifest Schema

File: `zerve_blocks/.block_manifest.json`

```json
{
  "last_updated": "2026-03-22T14:30:00",
  "blocks": [
    {
      "file": "01_load_statcan_data.py",
      "block_name": "load_statcan_data",
      "number": "01",
      "dag_role": "critical-path",
      "status": "baseline",
      "sha256_16": "abc123def456...",
      "prompts_sent": ["block01_v1.md"],
      "notes": ""
    }
  ]
}
```

Status transitions: `baseline -> prompt-sent -> agent-updated -> reviewed`
A block can cycle back to `prompt-sent` if review finds issues.

## Prompt History Format

Directory: `zerve_blocks/.prompts/`
One file per prompt attempt, versioned by block and attempt number.

```markdown
# Prompt: <block_name> — <goal summary>
Date: YYYY-MM-DD
Version: N
Target block(s): <block_name(s)>
Bead: <bead-id if applicable>

## Prompt (paste this into Zerve agent)

<the prompt text>

## Acceptance Criteria

- [ ] criterion 1
- [ ] criterion 2

## Result

Status: pending | accepted | needs-followup
Agent response notes: <filled in after ingest>
What worked: <what the agent got right>
What to fix: <gaps for next version, if any>
```

## Handling Block Changes

When the user reports that the Zerve agent added, deleted, merged, or split blocks:

- **Added**: Create new local file, add manifest entry, assign next number or
  user-specified number. Update DAG notes.
- **Deleted**: Mark manifest entry status = `deleted`, remove local `.py` file.
  Do NOT renumber other blocks.
- **Merged**: Treat as delete of source blocks + update of target block.
- **Split**: Treat as delete of source block + add of new blocks.
- **Renamed**: Update manifest `file` and `block_name` fields, rename local file.

After any structural change, re-run Phase 1 to rebuild the manifest.

## Rules

- **Token budget is real money.** Every prompt costs $2-4 in the user's Zerve token
  budget. Prompts MUST be as short as possible while still being precise. No
  preamble, no explanations of why a change matters, no repeated context the agent
  already has. Bullet points over paragraphs. Cut every word that doesn't change
  the agent's output.
- Prompts should reference Zerve concepts (blocks, canvas, DAG) not local repo paths.
- When the user says "the agent gave me this" or pastes code, treat it as Phase 3.
- When blocks are added/deleted, re-run Phase 1 before anything else.
- Track all prompts — prompt history shows iteration and is itself evidence of
  methodology for the hackathon.
- Keep prompts focused: one improvement per prompt is better than a kitchen-sink
  prompt, unless changes are tightly coupled.
