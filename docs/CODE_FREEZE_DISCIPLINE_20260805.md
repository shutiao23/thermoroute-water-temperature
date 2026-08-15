# Source-hash freeze discipline (in force until the one-time opening completes)

**Date in force:** 2026-08-05
**Lifts when:** the one-time 2021–2023 opening has completed and its receipts are validated.
**Status:** BINDING.

---

## 1. The rule

> **No byte may change under `src/`, `scripts/`, `tests/`, `protocols/`,
> `pyproject.toml`, `requirements*.txt`, or `.github/workflows/` until the opening
> is complete.**

Not a typo fix. Not a docstring. Not a lint autofix. Not "while I'm in there".
Not a new test.

## 2. Why the rule is absolute

`DEFAULT_SOURCE_PATTERNS` in [`src/thermoroute/repro.py`](../src/thermoroute/repro.py)
defines the hashed set:

```python
DEFAULT_SOURCE_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "tests/**/*.py",
    "protocols/**/*.json",
    "protocols/**/*.md",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock*.txt",
)
```

Every training receipt embeds `source_sha256 = source_tree_hash(root)`, and every
downstream consumer re-derives that hash from the **live working tree** and fails
closed on mismatch — for example in
[`src/thermoroute/model_suite.py`](../src/thermoroute/model_suite.py):

```python
if identity["source_sha256"] != source_tree_hash(root):
    raise ModelSuiteError("Stage-9 completion receipt is stale for the run or current source")
```

The same guard exists for Stage-16, for the development contract, and in
`development_replay.py`, `development_controls_gate.py`, `stage09_parallel.py` and
`stage09b_precompute.py`.

**Consequence: any edit to a hashed path retroactively invalidates all four training
receipts and forces a complete retrain of Stage-09, 09b, 16 and 25 — roughly two
days of wall clock.**

This has already been paid five times, for changes that could not affect any number:

| Commit | Nature | Cost |
|---|---|---|
| `a8dbcee` | raise worker caps to 96 | one lineage |
| `61eb21b` | unified numerical-policy role cap v2 | one lineage |
| `2e7e6c0` | LightGBM training `n_jobs` follows policy | one lineage |
| `64839c8` | Stage-09b completion gate follows policy cap | one lineage |
| `1e517b7` | 09b gate float slack on bridge report | one lineage |

Five full retrains, all of them about thread counts and gate thresholds. Zero
scientific content.

## 3. What IS safe to change right now

These paths are **not** hashed and may be edited freely, including while a chain is
running:

- `paper/**` — manuscript, SI, figures, figure renderers, figure specs
- `docs/**` — all design notes, decision records, audits
- `ops/**` — launchers, watchers, monitors
- `README.md`
- `outputs/**`, `data_usgs/**`

Verified by inspection of `DEFAULT_SOURCE_PATTERNS`: none of these match any pattern.

### 3.0 Never use git to restore a file in a tree with uncommitted work

`git checkout -- <path>`, `git restore <path>`, `git stash`, `git reset` and
`git clean` all discard uncommitted content. In a worktree where work is in
progress — especially one with concurrent writers — they destroy it silently
and without a prompt.

This happened twice on 2026-08-05/06:

| Operation | Effect |
|---|---|
| `git stash push --keep-index`, run to get a lint baseline | reverted **43 files** of uncommitted work across the shared worktree |
| `git checkout -- tests/test_repro.py`, run to undo a two-line probe | discarded the two-tier-hash test updates, **112 lines** |

Both were recovered — the first from the stash, the second from the dropped
stash's dangling commit — but recovery was luck, not design. Nothing had been
staged, so the object database held these blobs only incidentally.

**Rule:** to make a temporary edit, copy the file aside first and copy it back:

```bash
cp path/to/file /tmp/file.bak      # not: git checkout -- path/to/file
# ... make the temporary edit, measure, ...
cp /tmp/file.bak path/to/file
```

Verify a restore by content, not by absence of complaint. When the change was
made to observe a hash, the hash returning to its prior value *is* the proof:

```
model_source_hash : fb03cbdd…  matches pre-edit baseline : True
harness_hash      : a8a4ebd5…  matches pre-edit baseline : True
```

### 3.1 One exception that is about processes, not hashes

**Do not edit a shell script that is currently executing**, even an unhashed one
under `ops/`. `bash` reads a script incrementally; editing it in place changes what
the running interpreter reads next and can silently truncate the chain. This
previously caused a silent post-Stage-25 chain exit.

Write a new file and switch over at the next restart instead.

## 4. Handling a defect discovered during the freeze

1. **Do not fix it.** Record it.
2. Write a decision record in `docs/` stating the defect, its measured magnitude, the
   cost of fixing it now, and the chosen disposition.
3. Add the fix to the queue in
   [`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md),
   so that **all** deferred fixes are paid for with a single lineage rather than one
   each.
4. If the defect makes the science wrong — as opposed to making a stage unavailable —
   escalate immediately; that is the only case where breaking the freeze is correct.

Worked example: [`STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`](STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md).

## 5. Why not just fix the hash scope now

Because removing `tests/**/*.py` from `DEFAULT_SOURCE_PATTERNS` is itself an edit to
`src/thermoroute/repro.py`, which is hashed. The remediation costs exactly one
lineage, and it must be paid **after** the opening, bundled with every other deferred
fix. See the remediation plan.

## 6. Checklist before any commit during the freeze

```bash
git diff --cached --name-only | grep -E '^(src/|scripts/|tests/|protocols/|pyproject\.toml|requirements)' \
  && echo "STOP: this commit touches a hashed path and will invalidate the training receipts" \
  || echo "safe: no hashed path touched"
```
