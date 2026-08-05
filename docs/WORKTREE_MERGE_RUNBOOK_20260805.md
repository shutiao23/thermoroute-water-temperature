# Worktree divergence: authority decision and merge runbook

**Date:** 2026-08-05
**Status:** PLANNED — execute only after the running chain (24 → 27 → 14 → 26) has exited.
**Risk if not done:** Stage-14 `--strict-environment` and Stage-26 claim validation operate against a tree whose identity no receipt binds.

---

## 1. The situation

Two git worktrees of one repository had drifted apart:

| Worktree | Branch | HEAD | Size | Role |
|---|---|---|---|---|
| `thermoroute-water-temperature` | `feat/route-a-completion` | `b0699a8` | 2.6 GB | manuscript, SI, figures |
| `thermoroute-water-temperature-multicore` | `feat/multicore` | `dc9669f` | 15 GB | the live chain, all receipts, all model artifacts |

Measured `source_tree_hash`:

| Tree | Hash |
|---|---|
| `feat/multicore` (live) | `bf9500aaf549de02e6abc0fe471147ef66903205ad1450f0a2285579e32539e9` |
| `feat/route-a-completion` (live) | `bdc11ad7a22edbfe1d332a142b8c9815810183a39331b4ae3533d19484e62f37` |

## 2. Authority decision

**`feat/multicore` is authoritative.**

All four training receipts bind `bf9500aa…`, verified directly:

| Receipt | `source_sha256` | Matches live multicore tree |
|---|---|---|
| `route_a_stage09_completion.json` | `bf9500aa…` | yes |
| `route_a_stage09b_completion.json` | `bf9500aa…` | yes |
| `route_a_stage16_completion.json` | `bf9500aa…` | yes |
| `route_a_stage25_completion.json` | `bf9500aa…` | yes |

The manuscript tree's `bdc11ad7…` identity is bound by **no receipt**. It cannot
become authoritative without a full retrain, so the merge must preserve
`bf9500aa…` exactly.

## 3. The divergence is disjoint — the merge is safe

Merge base: `2440230` (`ops: bind launcher/watcher to bdc11ad7`).

**`feat/multicore` ahead by 4 commits, hashed paths only:**

```
dc9669f  ops: bind launcher/watcher to bf9500aa (bridge report slack fix)
1e517b7  fix: 09b gate compares bridge report with the documented float slack
963dae3  ops: bind launcher/watcher to dd834e25 (09b gate fix)
64839c8  fix: Stage-09b completion gate and release verifier follow policy cap
```

Touching exactly five files:
`scripts/verify_release.py`, `src/thermoroute/development_controls_gate.py`,
`src/thermoroute/repro.py`, `tests/test_manifest_release.py`,
`tests/test_stage09b_completion.py`.

**`feat/route-a-completion` ahead by 5 commits, unhashed paths only:**

```
b0699a8  chore: commit preopen figure redraw artifacts for fig01 and SI
eb17643  fix(paper): resolve text/line overlaps in fig01 pre-opening concept svg
d571fe6  docs: experiment status note for mentor review
39372b2  Merge branch 'feat/multicore' into feat/route-a-completion
eb9ff69  docs(bib): fix dataRetrieval author spellings and official metadata
```

Touching `paper/**`, `docs/**`, `paper/references.bib` — 43,595 insertions, none of
them in a hashed path.

**The two change sets modify no file in common.** A three-way merge therefore
resolves without conflict and keeps `feat/multicore`'s version of all five hashed
files, so `source_tree_hash` remains `bf9500aa…`.

## 4. Runbook

### Precondition

```bash
pgrep -f "24_freeze_model_suite|27_verify_development_replay|14_manifest|26_validate_claims|chain_stage09_remaining" \
  && echo "STOP — chain still running, do not merge" \
  || echo "safe to proceed"
```

Also stop the watcher before touching the tree:

```bash
pkill -f monitor_chains.sh
```

### Step 1 — commit the manuscript-tree work

```bash
cd /home/lzq/workspace/parttime/thermoroute-water-temperature
git status --short
git add docs/ paper/
git commit
```

### Step 2 — merge into the authoritative branch

```bash
cd /home/lzq/workspace/parttime/thermoroute-water-temperature-multicore
git merge feat/route-a-completion
```

Expected: clean merge, no conflicts.

### Step 3 — the gate that decides success

```bash
cd /home/lzq/workspace/parttime/thermoroute-water-temperature-multicore
python3 -c "import sys;sys.path.insert(0,'src');from thermoroute.repro import source_tree_hash;print(source_tree_hash('.'))"
```

**Must print `bf9500aaf549de02e6abc0fe471147ef66903205ad1450f0a2285579e32539e9`.**

If it prints anything else, the merge touched a hashed path. Do not proceed —
`git merge --abort` (or reset to `dc9669f`) and re-examine.

### Step 4 — confirm the receipts still validate

```bash
python3 - <<'PY'
import json, pathlib, sys
sys.path.insert(0, 'src')
from thermoroute.repro import source_tree_hash
live = source_tree_hash('.')
for f in ('route_a_stage09_completion.json', 'route_a_stage09b_completion.json',
          'route_a_stage16_completion.json', 'route_a_stage25_completion.json'):
    d = json.loads(pathlib.Path('outputs/models', f).read_text())
    def dig(o, depth=0):
        if depth > 4 or not isinstance(o, dict): return None
        if isinstance(o.get('source_sha256'), str): return o['source_sha256']
        for v in o.values():
            r = dig(v, depth + 1)
            if r: return r
    print(f, dig(d) == live)
PY
```

All four must print `True`.

### Step 5 — retire the divergence

Fast-forward the manuscript branch so only one line of history continues:

```bash
cd /home/lzq/workspace/parttime/thermoroute-water-temperature
git merge --ff-only feat/multicore
```

From this point there is one authoritative branch. Do not resume parallel
development on two branches that both touch hashed paths.

## 5. Standing rule afterwards

The manuscript worktree may continue to be used for `paper/**` and `docs/**` work,
because those paths are not hashed. It must never again originate a change under
`src/`, `scripts/`, `tests/` or `protocols/`. See
[`CODE_FREEZE_DISCIPLINE_20260805.md`](CODE_FREEZE_DISCIPLINE_20260805.md).
