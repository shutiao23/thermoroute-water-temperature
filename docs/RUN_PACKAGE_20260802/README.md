# ThermoRoute A100 / high-core-CPU run package — 2026-08-02

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Branch | `feat/route-a-completion` |
| Source SHA-256 (post-fix) | `a4b174e10d2ffae192a4411b3b29c5c00b52cd355e3cf409b0fa9dfe2e276002` |
| Previous source (historical only) | `19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` |
| Purpose | Execute the new Stage-09 → 09b → 16 → 25 lineage on a target machine (A100 GPU box or high-core CPU host) |
| Status | **NOT A COMPLETION RECEIPT.** This package prepares execution; it does not authorize opening or claims |

## 1. Hardware guidance (GPU versus CPU)

The Route-A **formal publication path is CPU-only by design** (cross-platform
determinism). `scripts/09_usgs_experiment.py` rejects non-CPU formal runs
(`formal_publication_candidate` requires `training_device == "cpu"`), and this
constraint is enforced throughout checkpoint, chronology, and gate validation.
A100 GPUs cannot accelerate the formal chain.

| Workload | Recommended hardware | Estimated wall time |
| --- | --- | --- |
| Stage-09 (9 models × 5 seeds, 438k rows) | 64–128-core CPU | 4–8 h (22 cores: 12–16 h) |
| Stage-09b (45 members, same scale) | 64–128-core CPU | 4–8 h |
| Stage-16 LSTM / Stage-25 external | 64–128-core CPU | 1–3 h each |
| Exploratory prototypes / hyperparameter sweeps | A100 (any device) | 1–3 h total |

If only an A100 box is available, run the formal chain on its host CPUs with a
high `--control-workers`/`PHASE2_09B_WORKERS` value; GPU resources are not used
by the formal gate but the machine's CPU count still determines throughput.
Do not pass `--device cuda` to any formal entrypoint; it will fail closed.

## 2. Machine prerequisites

- Linux (x86-64), Python 3.12, ~64 GB RAM recommended (30 GB minimum; the old
  22-core host OOM'd near the end of Stage-09 with ~24 GiB single-process RSS —
  reduce `PHASE2_09B_WORKERS`/`STAGE09_CONTROL_WORKERS` if memory is tight).
- Disk: ≥ 50 GB free for outputs, checkpoints, and bundles.

## 3. Environment setup (target machine)

```bash
# 1. Clone and checkout the exact branch tip
git clone https://github.com/shutiao23/thermoroute-water-temperature.git
cd thermoroute-water-temperature
git checkout feat/route-a-completion

# 2. Create the Python environment
python3.12 -m venv .venv-route-a
source .venv-route-a/bin/activate
pip install --upgrade pip

# 3. Install exact locked dependencies (hash-locked file included in repo)
pip install -r requirements-lock-py312-hashed.txt

# 4. Verify the environment matches the frozen source identity
python -c "import torch, pandas, numpy, lightgbm, pyarrow, scipy, statsmodels; \
print(torch.__version__, pandas.__version__, numpy.__version__, lightgbm.__version__)"

# 5. Verify the source hash matches this package
PYTHONPATH=src python - <<'EOF'
from thermoroute.repro import source_tree_hash
h = source_tree_hash(".")
print("source_tree_hash:", h)
assert h == "a4b174e10d2ffae192a4411b3b29c5c00b52cd355e3cf409b0fa9dfe2e276002", "source mismatch"
print("OK: matches run package")
EOF
```

If `requirements-lock-py312-hashed.txt` fails on the target image (e.g.
PyTorch wheel availability), the approved fallback is
`requirements-lock.txt`, then `requirements.txt`; record the final environment
fingerprint (`outputs/runs/<run_id>/run.json` captures it automatically).

## 4. Preflight checks (must all pass)

```bash
# No live process, no contested locks
pgrep -f '09_usgs_experiment|09b_development_controls|phase2_watch' || echo "clean"

# Focused regression suite from the repair (T1–T5 and full suite)
python -m pytest -q tests/test_development_controls.py tests/test_stage09b_precompute.py
python -m pytest -q tests/            # full suite, ~17 min on 22 cores

# Frozen data artifacts present
ls -la data_usgs/panel_usgs_120v2.parquet data_usgs/station_registry_v1.csv \
      data_usgs/frozen_panel_v1.json data_usgs/development_predictor_bridge_v1.json
```

## 5. Execution sequence (guarded entrypoints only)

The order is mandated by `docs/STAGE09B_PROTECTED_CHANGE_WORK_ORDER.md` §7.
The watcher pin file `ops/stage09/phase2_watch.env` is already updated with the
new source hash and an empty `EXPECTED_STAGE09_RUN_ID`.

```bash
# Step A: new guarded Stage-09 (creates new content-addressed run under a4b174e1…)
# Memory-sensitive hosts: STAGE09_CONTROL_WORKERS=4
export STAGE09_CONTROL_WORKERS=6
bash ops/stage09/start_stage09.sh

# Step B: independently validate the new receipt
PYTHONPATH=src python - <<'EOF'
import json
from pathlib import Path
from thermoroute.repro import source_tree_hash
r = json.load(open("outputs/models/route_a_stage09_completion.json"))
assert r["status"] == "PASS_FORMAL_STAGE09_COMPLETE"
assert r["run_identity"]["source_sha256"] == source_tree_hash(".")
assert r["confirmation_outcomes_requested_or_read"] is False
print("new Stage-09 receipt OK:", r["run_id"])
EOF

# Step C: bind watcher to the verified run ID, then start Phase 2
# (edit ops/stage09/phase2_watch.env: EXPECTED_STAGE09_RUN_ID=<new run id>)
bash ops/stage09/start_phase2_watch.sh

# Step D: Phase 2 proceeds automatically:
#   Stage-09b (45 members) → Stage-16 LSTM → Stage-25 external suite
# Continue only after each completion receipt validates.
```

Do NOT:
- resume `a930214d93fb7bdca83e` (failed pre-member 09b) or `bb02498a...` (void);
- pass `--device cuda` to any formal entrypoint;
- manually create receipts, work orders, or manifests;
- touch `outputs/models/route_a_stage09_completion.json` by hand;
- request, read, or open any 2021–2023 outcome without a separate authorization.

## 6. Post-run verification checklist

| Artifact | Acceptance |
| --- | --- |
| New Stage-09 receipt | `PASS_FORMAL_STAGE09_COMPLETE`, source bound to `a4b174e1…`, outcome flag false |
| Stage-09b receipt | `PASS`, 9 arms × 5 seeds = 45 member receipts, no outcome access |
| Stage-16 receipt | global LSTM closure + transfer folds |
| Stage-25 receipt | external pooled suite closure |
| Full test suite | `pytest -q tests/` passes on the target machine |
| Numeric tolerance | frozen parity thresholds from `requirements-lock*` and CI policy |

## 7. GPU evaluation notes (for the exploratory track only)

The exploratory track (Route-B prototypes, synthetic identifiability tests,
comparator pilots) may use A100. When doing so:

```bash
python scripts/09_usgs_experiment.py --device cuda --exploratory   # never formal
```

Any exploratory output must remain labeled
`[pending — 探索期数据，不可写入结论]` and cannot be promoted into the formal
receipt chain.

## 8. What this package does NOT do

- It does not authorize the one-time 2021–2023 opening (`P0-A3` still requires
  explicit approval after the development chain closes).
- It does not close FAIR/rights/DOI (`P0-F` remains external-blocked).
- It does not change Route A's descriptive-only inferential status
  (`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` remains permanent for the five rows).
- Route B (confirmatory) remains a separate, owner/custodian-signed decision
  (`docs/ROUTE_B_OWNER_CUSTODIAN_DECISION_FORM.md` D01–D13 / G01–G16).
