# B3 — Number Placeholder Map (ThermoRoute conventional paper)

- **Branch:** `feat/conventional`
- **Date:** 2026-08-06
- **Mode:** read-only audit (no repository file was modified during the audit; this is the audit record)
- **Scope of fill targets:** `paper/ThermoRoute_paper.md` (1335 lines, current conventional rewrite)
- **CSV producer inspected:** `scripts/conventional_holdout_2021_2023.py` → `src/thermoroute/conventional_score.py::compute_metrics_long` → `outputs/conventional/holdout_metrics_2021_2023.csv` (columns `model, horizon, metric, value, n`). The CSV was **not yet present** at audit time (holdout run in progress; `run_2021_2023.log` shows fetching up to 40/120 stations).
- **Cross-checks:** `paper/agu_submission/ThermoRoute_WRR.tex` (the AGU build mirrors the markdown 1:1: every `<<TOKEN>>` becomes `\ph{TOKEN}`); model-name constants in the run script's `TEMPORAL_BUNDLES` / `LSTM_BUNDLE` / `EXTERNAL_BUNDLES` dicts and `conventional_score.py::baseline_frames`.

> **Headline counts**
> - Total placeholders in the main paper: **253** = **238** `<<...>>` numeric/cell placeholders (Section 4.6) + **15** `[...]` external-input placeholders.
> - Of the 238 `<<...>>`: **198** direct-from-CSV, **38** derived, **2** external/separate-computation.
> - Model-naming inconsistencies: **2** (ablation case mismatch; Table 4.8 omits `DampedPriorOnly` that the CSV emits).
> - D4 residual apparatus terms found: **0 in main-paper prose**; **heavy in SI** — see §6.

---

## 1. Placeholder inventory (main paper)

Two placeholder syntaxes are used. No `XXX`, `TBD`, `____`, `{{...}}`, `??`, `TODO`, or empty table cells exist in the main paper (verified). The development-period numbers (e.g. +0.251, 0.560 °C, 2.217) are already concrete and are **not** placeholders.

### 1.1 `<<...>>` holdout placeholders — 238 distinct (239 occurrences; +1 is a prose example at L1275)

Token grammar: `<<{METRIC}_{MODEL}_{hN}>>` for the per-cell tables, plus two whole-table tokens. All live in §4.6 (lines 822–910). The run script and `compute_metrics_long` are the only producers of the CSV these map to.

#### 1.1.1 Table 4.6 — paired comparisons on the held-out window (20 placeholders, lines 839–843)

Median station ΔRMSE (eq. 9, °C), whole-HUC2 cluster-bootstrap 95% CI, ThermoRoute win rate. **These are NOT in the CSV** (CSV is pooled per model; no station-level pairing, no cluster bootstrap, no win rate). See §5 blocker.

| # | Row (line) | Placeholders | Source class |
|---|---|---|---|
| 1 | ThermoRoute vs. damped persistence, 1 d (L839) | `<<dRMSE_DP_h1>>` `<<CIlo_DP_h1>>` `<<CIhi_DP_h1>>` `<<win_DP_h1>>` | Derived, CSV-insufficient |
| 2 | ThermoRoute vs. damped persistence, 3 d (L840) | `<<dRMSE_DP_h3>>` `<<CIlo_DP_h3>>` `<<CIhi_DP_h3>>` `<<win_DP_h3>>` | Derived, CSV-insufficient |
| 3 | ThermoRoute vs. damped persistence, 7 d (L841) | `<<dRMSE_DP_h7>>` `<<CIlo_DP_h7>>` `<<CIhi_DP_h7>>` `<<win_DP_h7>>` | Derived, CSV-insufficient |
| 4 | ThermoRoute vs. LightGBM, 3 d (L842) | `<<dRMSE_LG_h3>>` `<<CIlo_LG_h3>>` `<<CIhi_LG_h3>>` `<<win_LG_h3>>` | Derived, CSV-insufficient |
| 5 | ThermoRoute vs. LightGBM, 7 d (L843) | `<<dRMSE_LG_h7>>` `<<CIlo_LG_h7>>` `<<CIhi_LG_h7>>` `<<win_LG_h7>>` | Derived, CSV-insufficient |

- `DP` = damped persistence (= CSV model `DampedPersistence`); `LG` = LightGBM.
- `dRMSE_*` = median across stations of (RMSE_ThermoRoute − RMSE_reference) on paired keys; `CIlo/CIhi` = whole-HUC2 cluster-bootstrap percentile interval; `win_*` = fraction of stations favouring ThermoRoute.
- Required computation: station-level paired RMSE per station + 10 000-draw whole-HUC2 cluster bootstrap. Source module candidates: `src/thermoroute/significance.py`. The CSV alone cannot produce these.

#### 1.1.2 Table 4.7 — primary models (108 placeholders, lines 857–874)

Six rows of models × 3 horizons × 6 cells. Token = `<<{METRIC}_{MODEL}_{h1|h3|h7}>>`.

- **Models (12 rows):** `Persistence`, `DampedPersistence`, `Climatology`, `LightGBM`, `LSTM`, `ThermoRoute`.
- **Metrics (6 cells/row):** `RMSE`, `MAE`, `bias`, `skill_vs_persistence`, `skill_vs_climatology`, `n`.
- Example row (L872): `<<RMSE_ThermoRoute_h1>>` `<<MAE_ThermoRoute_h1>>` `<<bias_ThermoRoute_h1>>` `<<skill_vs_persistence_ThermoRoute_h1>>` `<<skill_vs_climatology_ThermoRoute_h1>>` `<<n_ThermoRoute_h1>>`.

Source mapping per model class (see §3/§4 for name + metric mapping):

| Model | RMSE/MAE/bias/n (12 cells) | skill_vs_persistence (3) | skill_vs_climatology (3) |
|---|---|---|---|
| LightGBM, LSTM, ThermoRoute | CSV direct (RMSE/MAE/BIAS rows + `n` col) | CSV direct (`SKILL_PERSISTENCE`) | CSV direct (`SKILL_CLIMATOLOGY`) |
| Persistence, DampedPersistence, Climatology | CSV direct (RMSE/MAE/BIAS + `n`) | **Derived** (baselines emit no SKILL rows) | **Derived** (baselines emit no SKILL rows) |

→ 36 direct + 36 direct + 18 direct + 18 derived = Table 4.7 totals: **90 direct, 18 derived**.

#### 1.1.3 Table 4.8 — one-factor ablations (108 placeholders, lines 880–897)

Six ablation rows × 3 horizons × 6 cells. Token = `<<{METRIC}_{MODEL}_{h1|h3|h7}>>`.

- **Models (paper labels, lowercase):** `tr-fixedkappa`, `tr-nodynamicprior`, `tr-nomoe`, `tr-norouter`, `tr-notcn`, `tr-unbounded`.
- **Metrics (6 cells/row):** same six as Table 4.7.
- All six are non-baseline learned models, so all 108 cells are **CSV direct** — BUT require the case/name mapping in §3 (CSV emits CamelCase `TR-fixedKappa` etc.).

#### 1.1.4 Whole-table placeholders (2, lines 900 & 907)

| Placeholder | Line | Table | Source (NOT the holdout_metrics CSV) |
|---|---|---|---|
| `<<interval_probability_2021_2023>>` | 900 | Table 4.9 — interval & probability behaviour | Probability-metrics computation: empirical 90% coverage, mean width, 3-quantile pinball mean, Brier + Brier skill, log loss, AUROC/AUPRC, ECE, calibration slope/intercept. Modules: `conformal.py`, `probability.py`, `probability_metric_erratum.py`; SI08. |
| `<<outcome_qc_2021_2023>>` | 907 | Table 4.10 — outcome QC | Outcome-QC computation: counts by station/variable/raw-qualifier/value-presence, multi-finite-series conflict counts, `{A}`-restricted sensitivity. Module: `outcome_qc.py`; SI12. |

### 1.2 `[...]` external-input placeholders — 15 (lines 3–1304)

Non-numeric, human/metadata inputs. The "Placeholder inventory" paragraph (L1268–1280) re-lists items 5–11 below; that re-listing is not a separate placeholder. Five of these span two or more lines (closing `]` on a later line).

| # | Placeholder | Line(s) | Input required | Source |
|---|---|---|---|---|
| 1 | `[AUTHOR LIST TO BE COMPLETED]` | 3 | Final author list, order, ORCIDs | `docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2 intake |
| 2 | `[AFFILIATION 1 TO BE COMPLETED — …]` | 9 | Dept/lab, institution, city, postcode, country | Author intake |
| 3 | `[AFFILIATION 2 TO BE COMPLETED — …]` | 11 | Add/delete affiliation lines to match author list | Author intake |
| 4 | `[CORRESPONDING AUTHOR TO BE COMPLETED — …]` | 13–14 | Verified name, ORCID, affiliation no., institutional e-mail | Author intake |
| 5 | `[TRAINING WALL-CLOCK TO BE RECORDED]` | 377 | End-to-end training wall-clock (not in run records) | Operator record / run log |
| 6 | `[INFERENCE COST TO BE RECORDED]` | 378 | Inference throughput (not in run records) | Operator record / run log |
| 7 | `[DATA DOI TO BE MINTED]` | 1200 | Dataset DOI (after rights review, §6.4) | Post-rights minting |
| 8 | `[DATA LICENCE TO BE ASSIGNED]` | 1201 | Data licence (after rights review) | Rights decision |
| 9 | `[SOFTWARE DOI TO BE MINTED]` | 1243 | Software archive DOI | Minting |
| 10 | `[RELEASE TAG TO BE ASSIGNED]` | 1244 | Source repo release tag | Release |
| 11 | `[REPOSITORY URL TO BE CONFIRMED]` | 1245 | Confirmed repository URL | Release |
| 12 | `[FUNDING TO BE COMPLETED — …]` | 1284–1285 | Each funder + award number, or explicit "no external funding" | Author intake |
| 13 | `[COMPUTATIONAL RESOURCES TO BE COMPLETED — …]` | 1287–1288 | Facility/facilities used for fitting | Author intake |
| 14 | `[COMPETING INTERESTS TO BE COMPLETED — …]` | 1295–1297 | Declaration from every author | Author intake |
| 15 | `[CREDIT ROLES TO BE COMPLETED — …]` | 1299–1304 | CRediT roles per author | Author intake (§2 intake form) |

### 1.3 Auxiliary / optional blank (not a `<<...>>` token)

- **L223** temporal-roles table, "Held-out test" row: the `Rows` cell is `—` and the `Observed WTEMP` cell is `observed`. These are intentional descriptive blanks, not fill tokens. If a concrete held-out row count is desired, it is derivable from `outputs/conventional/panel_2021_2023.parquet` once the panel is assembled. Not counted in the 253.

---

## 2. Source classification summary (238 `<<...>>`)

| Class | Count | Description |
|---|---:|---|
| **CSV direct** | 198 | Read directly from `holdout_metrics_2021_2023.csv` with name/metric mapping (§3, §4). Tables 4.7 (90) + 4.8 (108). |
| **Derived (from CSV)** | 18 | Table 4.7 baseline skill cells (Persistence/DampedPersistence/Climatology × 3 horizons × 2 skill metrics). Baselines emit no SKILL rows in the CSV; must be computed from CSV RMSE rows (§5). |
| **Derived, CSV-insufficient** | 20 | Table 4.6 paired ΔRMSE / cluster-bootstrap CI / win rate. Requires station-level pairing + HUC2 cluster bootstrap; the pooled CSV cannot supply these (§5). |
| **External / separate computation** | 2 | `<<interval_probability_2021_2023>>`, `<<outcome_qc_2021_2023>>`. Not from the holdout_metrics CSV. |
| **Total** | **238** | |

External-input `[...]` placeholders: **15** (author/metadata/DOI/licence — §1.2).

---

## 3. Model naming mapping

CSV `model` column values come from the run script's bundle dicts (`TEMPORAL_BUNDLES`, `LSTM_BUNDLE`, `EXTERNAL_BUNDLES`) and `conventional_score.py::baseline_frames` (keys `Persistence`, `DampedPersistence`, `Climatology`). The paper/tex table labels and placeholder tokens must be matched to these.

| Paper Table 4.7 label | Placeholder token model part | CSV `model` value | §3.2 display name | Match? |
|---|---|---|---|---|
| Persistence | `Persistence` | `Persistence` | Persistence | ✅ |
| DampedPersistence | `DampedPersistence` | `DampedPersistence` | Damped persistence | ✅ |
| Climatology | `Climatology` | `Climatology` | Seasonal climatology | ✅ |
| LightGBM | `LightGBM` | `LightGBM` | LightGBM | ✅ |
| LSTM | `LSTM` | `LSTM` | global LSTM | ✅ |
| ThermoRoute | `ThermoRoute` | `ThermoRoute` | ThermoRoute | ✅ |

| Paper Table 4.8 label | Placeholder token model part | CSV `model` value | §3.2 display name | Match? |
|---|---|---|---|---|
| tr-fixedkappa | `tr-fixedkappa` | `TR-fixedKappa` | `TR-fixedKappa` | ⚠️ **case mismatch** |
| tr-nodynamicprior | `tr-nodynamicprior` | `TR-noDynamicPrior` | `TR-noDynamicPrior` | ⚠️ **case mismatch** |
| tr-nomoe | `tr-nomoe` | `TR-noMoE` | `TR-noMoE` | ⚠️ **case mismatch** |
| tr-norouter | `tr-norouter` | `TR-noRouter` | `TR-noRouter` | ⚠️ **case mismatch** |
| tr-notcn | `tr-notcn` | `TR-noTCN` | `TR-noTCN` | ⚠️ **case mismatch** |
| tr-unbounded | `tr-unbounded` | `TR-unbounded` | `TR-unbounded` | ⚠️ **case mismatch** |

### Inconsistencies found

1. **Ablation case mismatch (Table 4.8).** The six ablation placeholder tokens use lowercase (`tr-fixedkappa`, `tr-nodynamicprior`, `tr-nomoe`, `tr-norouter`, `tr-notcn`, `tr-unbounded`), but the CSV `model` column emits CamelCase `TR-fixedKappa`, `TR-noDynamicPrior`, `TR-noRouter`, `TR-noMoE`, `TR-noTCN`, `TR-unbounded` (identical to the §3.2 display names). **The filler must match case-insensitively** (token `tr-fixedkappa` ≡ CSV `TR-fixedKappa`). Note this also makes Table 4.8 internally inconsistent with §3.2, which uses the CamelCase forms — a paper-side style fix to consider.
2. **`DampedPriorOnly` surplus in CSV.** The run script scores seven architecture controls (`DampedPriorOnly` plus the six above) because `skip_ablations=False` includes `DampedPriorOnly` in `temporal_models`. Table 4.8 lists only six ablations (it omits `DampedPriorOnly`, which §4.3 reports separately as "the damped anchor alone"). **The CSV will contain `DampedPriorOnly` rows with no placeholder.** Filler should ignore them for Table 4.8 (or wire them to the §4.3 prose, which has no placeholder).
3. **External `-ext` models surplus in CSV.** With `skip_external=False` the CSV also emits `ThermoRoute-ext`, `LSTM-ext`, `LightGBM-ext` (external 30-site cohort). **No placeholder in Tables 4.7/4.8 references these**; they belong to SI13 / the external arm. Filler should ignore them for the main tables.
4. **Pooled rows surplus in CSV.** `compute_metrics_long` also writes `horizon="pooled"` rows for every model. **No placeholder references pooled** (all tokens are `_h1/_h3/_h7`). Filler should select `horizon ∈ {1,3,7}` only.

---

## 4. Metric naming mapping

The initial requirements state CSV `metric ∈ {RMSE, MAE, BIAS, skill_vs_persistence, skill_vs_climatology}`, but the actual producer (`compute_metrics_long`) writes uppercase / underscore metric names and an extra count metric. The placeholder token prefixes use a different casing. Mapping required:

| Placeholder token prefix | CSV `metric` value | CSV `n` column | Notes |
|---|---|---|---|
| `RMSE_` | `RMSE` | `n` (key count) | direct |
| `MAE_` | `MAE` | `n` | direct |
| `bias_` | `BIAS` | `n` | ⚠️ case: token `bias` vs CSV `BIAS` |
| `skill_vs_persistence_` | `SKILL_PERSISTENCE` | `n` | ⚠️ name+case mismatch; only emitted for non-baseline models |
| `skill_vs_climatology_` | `SKILL_CLIMATOLOGY` | `n` | ⚠️ name+case mismatch; only emitted for non-baseline models |
| `n_` | (not a metric) | `n` | the `n` column value for that model×horizon (same on every metric row) |
| — (no placeholder) | `N_SKILL` | — | common-key count used for skill; **no placeholder consumes it** |

CSV metric semantics (from `compute_metrics_long`): RMSE/MAE/BIAS are computed **pooled over all common keys** for the model×horizon (not per-station-then-median); skill = `1 − RMSE_model/RMSE_baseline` computed on the keys common to model+Persistence+Climatology, i.e. a **pooled** skill (ratio of pooled RMSEs).

---

## 5. Critical caveats / fill blockers

These must be resolved before/during filling; several are substantive, not cosmetic.

1. **Pooled vs station-median (substantive).** Table 4.7/4.8 header (L845–848) states *"Station-median metrics per model and lead"*. The CSV produces **pooled** RMSE/MAE/BIAS and **pooled** skill (ratio of pooled RMSEs), not station-medians. Filling the Table 4.7/4.8 cells directly from the CSV would therefore contradict the table caption. Either (a) the CSV producer must be changed to emit station-median metrics, or (b) the caption must be reworded to "pooled". The development-period comparators in the paper (e.g. +0.251, +0.038) are station-median skills, so a pooled held-out skill is **not directly comparable** to them. Flag for the author.
2. **Baseline skill gap (18 derived cells).** `compute_metrics_long` puts `Persistence`, `DampedPersistence`, `Climatology` in `baseline_names` and emits **only RMSE/MAE/BIAS** for them — no `SKILL_*` rows. Yet Table 4.7 has `skill_vs_persistence` and `skill_vs_climatology` cells for all six models including the three baselines. The 18 baseline skill cells (3 baselines × 3 horizons × 2 skill metrics) must be **derived from CSV RMSE rows**:
   - Persistence: `skill_vs_persistence = 0` (definitional); `skill_vs_climatology = 1 − RMSE_Persistence / RMSE_Climatology`.
   - DampedPersistence: `skill_vs_persistence = 1 − RMSE_DampedPersistence / RMSE_Persistence`; `skill_vs_climatology = 1 − RMSE_DampedPersistence / RMSE_Climatology`.
   - Climatology: `skill_vs_persistence = 1 − RMSE_Climatology / RMSE_Persistence`; `skill_vs_climatology = 0` (definitional).
   (If caveat 1 moves everything to station-median, these derivations must use station-median RMSEs instead.)
3. **Table 4.6 is CSV-insufficient (20 cells).** The CSV gives one pooled RMSE per model×horizon. Table 4.6 needs the **median across stations** of the paired ΔRMSE, a **whole-HUC2 cluster-bootstrap 95% interval**, and a **ThermoRoute win rate** — none of which is in the CSV. A separate station-level paired + clustered computation is required (candidate: `src/thermoroute/significance.py`; the development-period analogues in §4.1/4.2 used the same clustered procedure). Without it, all 20 Table 4.6 placeholders are unfillable from the named CSV.
4. **Skill denominators / common-key set.** CSV skill uses the keys common to (model, Persistence, Climatology); the `n` column on a skill row is the model's own key count, while the count actually used for skill is the separate `N_SKILL` metric (no placeholder). If the paper wants the skill denominator count displayed, it is `N_SKILL`, not `n`.
5. **Horizon dtype.** CSV `horizon` is written as `int` for 1/3/7 and the literal string `"pooled"` for pooled rows. Filler must coerce to int and exclude `"pooled"`.
6. **Whole-table tokens (2) need their own producers.** `<<interval_probability_2021_2023>>` and `<<outcome_qc_2021_2023>>` are not derivable from `holdout_metrics_2021_2023.csv`; they require the probability-metrics and outcome-QC pipelines (§1.1.4). These are tracked separately (SI08 / SI12).

---

## 6. D4 residual language list (pre-registration / sealed / apparatus)

**Main paper (`paper/ThermoRoute_paper.md`):** prose is **clean**. No `seal`, `sealed`, `reseal`, `opening`, `one-time`, `receipt`, `apparatus`, `confirmatory`, `preregistration`, `prelabel`, `chronology`, `Route A/B`, or `frozen claim` appears in reader-visible text. The only `PRE`/`POST` occurrences are **internal HTML-comment figure-anchor tags** (`<!-- FIGURE_ANCHOR id=… state=PRE|POST … -->`, L157/209/278/381 and others) used by the figure renderer to mark development- vs held-out-period figures. These are scaffolding comments, not prose; whether they should be purged is a low-priority style decision (the conventional paper's figure-period semantics are legitimate, only the `PRE`/`POST` *vocabulary* is apparatus-flavoured).

**SI (`paper/si/SI*.md` + `README.md`):** heavy residual apparatus vocabulary. Classified below. "Legit" = retained conventional meaning (frozen weights/panel/inputs; MoE/regime gate; QC plausibility gate; `pre-` prefixes like pre-acquisition/pre-attrition/pre-CQR) — **not** flagged for removal.

### 6.1 SEAL / SEALED / RESEAL (apparatus) — 6 hits
- `SI02_information_boundary.md:127` — "the seals, ledgers, and chronology records are repository-internal"
- `SI04_protocol_and_amendments.md:5` — "does not reseal"
- `SI04_protocol_and_amendments.md:21` — "`*_seal_*.json` files are seal references, not a license to …"
- `SI04_protocol_and_amendments.md:49` — "gate as a post-seal reproducibility strengthening step"
- `SI04_protocol_and_amendments.md:60` — `protocol_seal_ref`
- `SI04_protocol_and_amendments.md:62` — `amendment_seal_ref`

### 6.2 OPENING / ONE-TIME OPENING (apparatus) — 8 hits
- `SI00_inventory.md:15` — "that opening or release closure occurred"
- `SI00_inventory.md:17` — "any result before opening-receipt verification"
- `SI00_inventory.md:40` — "opening, coverage, spatial and QC receipts"
- `SI00_inventory.md:125` — "trusted scorer inside the one-time"
- `SI00_inventory.md:126` — "opening, not from that stage"
- `SI01_cohort_and_registry.md:60` — "A post-opening count …"
- `SI02_information_boundary.md:99` — "There is one logical opening and one fixed request ledger"
- `SI04_protocol_and_amendments.md:66` — `opening_ref`

### 6.3 RECEIPT (apparatus) — pervasive: 10 filenames + ~40 content hits
- **Filenames (10):** `SI06_formal_five_rows_RECEIPT.md`, `SI07_all_model_scores_RECEIPT.md`, `SI08_probability_metrics_RECEIPT.md`, `SI09_development_controls_RECEIPT.md`, `SI10_temporal_coverage_RECEIPT.md`, `SI11_spatial_sensitivity_RECEIPT.md`, `SI12_qc_qualifiers_RECEIPT.md`, `SI13_external_history_arm_RECEIPT.md`, `SI14_missingness_failures_RECEIPT.md`, `SI15_reproduction_hashes_RECEIPT.md`.
- **Content hits (representative):** SI00:5,16,17,26,33,37,38,39,40,41,44,76,82,98; SI01:7,48,56; SI02:9,38; SI03:7,75; SI04:8,18,67; SI07:5,11,58; SI08:21,31,82; SI09:5,9,15,17; SI11:11; SI13:11,13; SI15:11; SI16:78; `README.md`:41–50 (the SI inventory table describes every SI06–SI15 as a `*_RECEIPT` projection). Theme: "receipt-bound predictions/evaluation", "test-window receipt", "completion receipt", "opening-receipt verification".

### 6.4 GATE (apparatus sense) — apparatus hits below; legit MoE/QC gates excluded
- Apparatus: `SI00:23` (gate routing), `SI00:36` (gate status), `SI00:75` (POST-gated), `SI00:82` (gated on the test-window receipt), `SI00:98` (receipt-gate rules), `SI00:99` (gates), `SI02:37–38` (compatibility gate / engineering-gate), `SI04:27` (outcome-free gate inputs), `SI04:49–50` (post-seal gate / PRE engineering-gate scope), `SI11:10` (balance gates), `SI11:24` (the gate itself reads), `SI11:27` (Gate), `SI11:34` (Gate thresholds), `SI11:47` (outcome-free gate), `SI11:51` (gate conditions), `SI11:54` (cluster gate), `SI11:71` (the gate itself), `SI11:80` (permanent gate status box), `SI12:7` (gate), `SI01:21` (minimum reportable-cluster gate — borderline eligibility gate).
- Legit (not flagged): `SI03:50` (mixture-gate = MoE), `SI16:92` (plausibility gate = QC range check −2…50 °C).

### 6.5 FROZEN (apparatus sense: frozen contract/claim/protocol/gate) — flagged hits; legit frozen weights/panel/inputs excluded
- Apparatus: `SI04:5` (frozen protocol sources … reseal), `SI04:17` (frozen metric contract), `SI08:33` (frozen probability contract), `SI08:56` (Rows tripping the frozen contract), `SI08:79` (frozen contract tests q05 ≥ q95), `SI08:80` (frozen contract forbids evaluation-time repair), `SI08:82` ("contract was frozen before the holdout labels were …" — sealed-logic), `SI02:37` (frozen … compatibility gate / engineering-gate), `SI09:14` (frozen suite + authorization chain).
- Legit (frozen cohort/panel/inputs/anchor/model/weights — **not** flagged): SI00:12,20,35,39,97; SI01:1,9,23,35; SI02:17,32,80,86; SI03:26,42,52,69; SI07:18,51,59; SI09:11,12; SI11:18; SI12:16; SI16:83,104; README:7,27,38,44. (Note SI01:9 "Frozen PRE geometry" is flagged under §6.6 for "PRE", not for "frozen".)

### 6.6 PRE / PRE-REGISTRATION vocabulary (apparatus) — SI prose
`PRE` is used throughout the SI as pre-registration apparatus vocabulary ("PRE manuscript", "PRE material/text/rule/configuration/source/narrative", "PRE/POST renderer", "PRE marker", `PRE_MATERIALIZED`), alongside `POST-gated`/`POST evidence chain`.
- `SI00:10,15,65,89,97,115`; `SI01:9,24,30`; `SI02:26,31,36`; `SI03:6,28,40,42,43,44,45,46,47,49`; `SI04:12,30,50`; `SI08:91` (pre-CQR — **legit**, not flagged).
- No literal `pre-registration` / `preregistration` string appears anywhere (main paper or SI).

### 6.7 ROUTE A / ROUTE B (apparatus study-family terms)
- `SI04:14` — `protocols/route_a_primary_v1.json` ("registered study family …")
- `SI04:17` — `protocols/route_a_probability_metric_erratum_v1.json`
- `SI04:18` — `protocols/route_a_temporal_coverage_policy_v1.json` ("receipt binding is required before any audit result")
- `SI11:10` — "Route B requires its own prelabel registry, balance gates"
- (Main paper: none.)

### 6.8 PRELABEL / CHRONOLOGY (apparatus module/registry terms)
- `SI11:10` — "prelabel registry" (with Route B).
- `chronology`: `SI00:15,36,65`; `SI02:127`; `SI04:1,14,53,55,69`; `SI15:8`. (Apparatus chronology/binding vocabulary.)

### 6.9 Auxiliary: SI `[pending computation]` cells
Separate from the main-paper `<<...>>` set, the SI files themselves contain `[pending computation]` table cells (the SI's own placeholder syntax) to be filled from the same CSV/computations: SI00(2), SI01(2), SI02(2), SI03(2), SI04(2), SI05(6), SI06(6), SI07(1), SI08(2), SI09(2), SI10(1), SI11(1), SI12(3), SI13(1), SI14(1), SI15(5), SI16(1). These are SI-side fill targets, not part of the 253 main-paper count, but they draw on the same `holdout_metrics_2021_2023.csv` and the probability/QC producers.

### 6.10 D4 term-count summary

| Term | Main paper (prose) | SI (prose + filenames) | Verdict |
|---|---:|---:|---|
| seal/sealed/reseal | 0 | 6 | SI residual |
| opening / one-time | 0 | 8 | SI residual |
| receipt | 0 | ~40 + 10 filenames | SI residual (largest) |
| gate (apparatus sense) | 0 | ~20 | SI residual |
| frozen (apparatus sense) | 0 | 9 | SI residual (legit frozen-weights uses excluded) |
| PRE / POST (apparatus vocab) | 0 (only HTML-comment figure tags) | ~25 | SI residual |
| Route A / Route B | 0 | 4 | SI residual |
| prelabel / chronology | 0 | ~8 | SI residual |
| pre-registration (literal) | 0 | 0 | clean |
| apparatus / confirmatory / blinded / unblind | 0 | 0 | clean |

**Bottom line:** the conventional main paper is apparatus-free in its prose; the residual apparatus language is concentrated in the SI (`paper/si/`), dominated by the `*_RECEIPT.md` file naming and the `receipt`/`PRE`/`opening`/`seal`/`gate`/`frozen contract`/`Route A` vocabulary. This is consistent with the in-progress `docs/APPARATUS_DELETION_PLAN.md` effort.

---

## 7. Files referenced (read-only)

- `paper/ThermoRoute_paper.md` (primary target, 1335 lines)
- `paper/agu_submission/ThermoRoute_WRR.tex` (confirms 1:1 `<<…>>`→`\ph{…}` and identical model labels)
- `scripts/conventional_holdout_2021_2023.py` (CSV producer; model-name constants; `compute_metrics_long` call at L353–355)
- `src/thermoroute/conventional_score.py` (`compute_metrics_long` L363–440; `baseline_frames` L298–338; metric names `RMSE/MAE/BIAS/SKILL_PERSISTENCE/SKILL_CLIMATOLOGY/N_SKILL`)
- `paper/si/SI*.md`, `paper/si/README.md` (D4 scan)
- `outputs/conventional/run_2021_2023.log`, `monitor.log` (run status: in progress, 40/120 stations fetched)
