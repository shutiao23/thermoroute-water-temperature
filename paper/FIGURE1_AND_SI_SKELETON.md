# Figure 1 and SI skeleton (R2-5; superseded visual layout)

> **Supersession note (2026-08-04).** This file preserves the earlier
> four-panel/PRE receipt-shell geometry for audit provenance.  It is no longer
> the visual specification for the paper figures.  The governing redraw,
> evidence gates, panel responsibilities, and removal of the old pending forest
> are defined in `paper/FIGURE_REDRAW_SPEC.md`.  Result cells remain receipt-gated;
> this supersession does not authorize opening or result filling.

**Status:** pre-opening design scaffold only.  
**Constraint:** no fabricated performance numbers. Every result cell is
`[pending — 探索期数据，不可写入结论]`. Values may be filled only from a
verified opening receipt, claim registry, inference/QC gates and POST evidence
manifest after Route-A authorization. A development-stage completion receipt
never authorizes a target-result cell by itself.

**Canonical manuscript:** `paper/ThermoRoute_paper.md`  
**Do not edit:** `src/`, `scripts/`, `tests/`, `protocols/` (source-hash tree).

---

## 0. Antagonistic fill rules

| Allowed now | Forbidden until receipt |
|---|---|
| Layout, captions, axis labels, equation forms | RMSE, MAE, skill, coverage %, Brier, etc. |
| Frozen design facts already fixed in manuscript text (e.g. ≤15 HUC2, gate thresholds) | Invented effect sizes, CI endpoints, p-values |
| Explicit `receipt-derived` placeholders | Copying exploratory 2019–2020 diagnostics into Figure 1D |
| Literature TODO checklist | Claiming a literature gap is “closed” without bib entry |

Placeholder token (use verbatim in every result cell):

```text
[pending — 探索期数据，不可写入结论]
```

---

## 1. Figure 1 — four-panel conceptual figure

**Working title:** Information boundary, bounded correction, spatial dependence, and receipt-gated effects.

**Intended placement:** after Methods §3–§4 (or early Results as a design figure).  
**Format:** 2×2 panel; vector preferred (PDF/SVG); AGU one-column or full-width TBD.

```text
┌─────────────────────────────┬─────────────────────────────┐
│ (a) Information boundary    │ (b) Bounded correction      │
│                             │     + non-safety warning    │
├─────────────────────────────┼─────────────────────────────┤
│ (c) Station–cluster         │ (d) Effect + CI             │
│     imbalance               │     (receipt-derived)       │
└─────────────────────────────┴─────────────────────────────┘
```

### 1a. Information boundary（信息边界）

**Purpose:** Show what the predictor may and may not see at issue time.

**Sketch elements (no numbers):**

- Timeline: calendar day `t` (issue) → targets at horizons `h ∈ {1, 3, 7}`.
- Left of `t` (allowed): observed WTEMP history through issue date; date-indexed
  Daymet/gridMET covariates with dates ≤ historical issue date; frozen
  climatology / damped anchor inputs.
- Right of `t` (forbidden): target-date WTEMP; horizon-specific future weather
  forecasts; any anticipatory covariate vintages.
- Caption callout: *date-indexed retrospective hindcast* — not operational
  as-issued availability, not ungauged prediction, not causal transport.

**Caption draft (EN):**

> (a) Issue-time information boundary for the Route-A hindcast. Predictors
> consume only left-looking history and date-indexed meteorology; they do not
> receive horizon-specific future forecasts or target outcomes.

**Result / metric cells:** none (design panel). Any accidental score annotation →
`[pending — 探索期数据，不可写入结论]`.

---

### 1b. Bounded correction equation + 非安全界警示

**Purpose:** State the algebraic point identity and the explicit *non-safety*
interpretation.

**Equation (canonical identity; symbols only — no scored error):**

\[
\hat{y}_{t+h}
=
A_{t+h}
+
\delta\,\tanh\!\Bigl(
\frac{P_{t+h}-A_{t+h}+r_{\theta,t+h}}{\delta}
\Bigr),
\qquad
\delta = \texttt{delta\_scale}
\]

where \(A_{t+h}\) is the frozen damped-persistence anchor,
\(P_{t+h}\) is the learned thermal proposal and \(r_{\theta,t+h}\) is the neural
residual. The total unrestricted displacement passed through `tanh` is
\(P_{t+h}-A_{t+h}+r_{\theta,t+h}\). With finite \(\delta\), the point forecast
lies in \([A_{t+h}-\delta,\,A_{t+h}+\delta]\). This matches the frozen
implementation path and the materialized SVG; the internal proposal is not a
separately identifiable physical state.

**Canonical configuration note (design, not performance):** manuscript freezes
`delta_scale = 1.0 °C` as an algebraic point bound relative to the anchor.

**Mandatory on-figure warning box (中/EN bilingual OK):**

> **非安全界 / NOT a safety bound.**  
> The \(\pm\delta\) envelope bounds *deviation from the anchor*, not absolute
> error, event-tail risk, interval width, post-shift behavior, or deployment
> safety. Do not read as a regulatory or operational guarantee
> (`LIMIT_NO_SAFETY_GUARANTEE`).

**Caption draft (EN):**

> (b) Bounded point correction about the damped-persistence anchor. The
> \(\tanh\) scaling yields a finite algebraic deviation from the anchor; it is
> not a truth-error or safety bound.

**Result cells:** none. Do not annotate panel (b) with RMSE or skill.

---

### 1c. Station–cluster imbalance（站点–cluster 不平衡）

**Purpose:** Visualize that daily rows ≠ independent samples and that the
inferential gate cannot pass on the frozen cohort.

**Sketch elements:**

- Map or bar chart of **stations per HUC2** for the frozen 120-site registry
  (15 HUC2 groups). Use registry-derived counts only; if a redraw is not yet
  scripted, leave bars as placeholders labeled `registry-derived (redraw TODO)`.
- Callouts already fixed in manuscript design text (**cohort geometry / gate
  inputs only — not RMSE, skill, or any scored performance**):
  - at most **15** reportable HUC2 clusters;
  - stations per HUC2 range **2–26** (pre-attrition design description);
  - largest-cluster share ≈ **21.7%** of stations;
  - inverse-Herfindahl effective cluster count ≈ **9.54**;
  - gate requires \(n_{\mathrm{clusters}}\ge 30\), effective fraction \(\ge 0.75\),
    largest share \(< 0.25\) → **necessarily fails** → permanent
    `descriptive (fixed cohort, few clusters)` / fixed-cohort descriptive only.
  - Any redraw-derived count that is *not* yet re-verified from the frozen
    registry must stay labeled `registry-derived (redraw TODO)` and must **not**
    be replaced by exploratory Stage-09 scores.

**Caption draft (EN):**

> (c) Station counts by HUC2 for the availability-enriched frozen cohort.
> With ≤15 clusters, the minimum-30-cluster cluster-structure caveat fails before any
> target-period outcome is viewed; formal wording remains descriptive only.

**Forbidden:** any panel annotation that implies national representativeness or
superpopulation coverage.

---

### 1d. Effect + CI（receipt-derived 占位）

**Purpose:** Reserve the exact five-comparison reporting layout so opening
receipts can drop in without redesign.

**Layout:** forest / dumbbell plot, five rows (frozen family):

| Row | Comparison | Horizon | Margin / ceiling | Effect (station-median ΔRMSE) | Cluster-bootstrap CI | Sign-flip p | Holm p | Verdict |
|---|---|---:|---|---|---|---|---|---|
| 1 | ThermoRoute − damped persistence | 1 d | 0.00 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` |
| 2 | ThermoRoute − damped persistence | 3 d | 0.00 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` |
| 3 | ThermoRoute − damped persistence | 7 d | 0.00 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` |
| 4 | ThermoRoute − LightGBM | 3 d | ceiling +0.05 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` |
| 5 | ThermoRoute − LightGBM | 7 d | ceiling +0.05 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` *(receipt-derived)* | `[pending — 探索期数据，不可写入结论]` |

**On-figure footnote (required):**

> All numeric cells are **receipt-derived** placeholders. Fill only from the
> verified opening / claim-ledger receipt. Exploratory 2019–2020 diagnostics are
> not admissible. Eligible Route-A wording remains fixed-cohort descriptive
> (`descriptive (fixed cohort, few clusters)`); p-values and CIs are
> approximate sensitivities, not superiority / national claims.

**Caption draft (EN):**

> (d) Predeclared five-comparison effects and cluster-bootstrap intervals.
> Values are withheld pending a verified receipt; the panel geometry is fixed
> so receipt fields can populate without redesign.

---

### Figure 1 production checklist

- [x] Draft self-contained vector layout matching 2×2 grid:
  `paper/agu_submission/figures/fig01_preopening_concept.svg` (PRE scaffold;
  XML well-formed; raster/page visual QA still pending)
- [ ] Panel (c) redraw script from `data_usgs/station_registry_v1.csv` (no scores)
- [ ] Panel (d) binder: map receipt JSON fields → table cells (fail-closed if missing)
- [ ] Bilingual 非安全界 callout locked in art file
- [x] AGU PRE scaffold path created under
  `paper/agu_submission/figures/fig01_preopening_concept.svg`; final filled figure,
  caption integration and submission render remain receipt/page-QA gated

---

## 2. Literature TODO — 2021–2026 gap fill

**Goal:** Close related-work gaps before Results text freezes.  
**Rule:** cite only after bib entry is web-verified; do not invent DOIs.

### 2.1 Must-address clusters

| Cluster | Status in repo today | TODO action |
|---|---|---|
| **Corona & Hogue 2025** (HESS ML stream-temperature review) | Present in `paper/references.bib` as `corona2025ml` | Wire into §1.1 + Discussion: evaluation metrics, leakage risks, baseline strength; contrast ThermoRoute’s common-key / clustered / pre-opening design |
| **中国流域 / China-basin river-temperature ML** | Sparse / incomplete relative to 2021–2026 literature | Search & add ≥2–3 verified 2021–2026 studies (Yangtze / Yellow / Pearl / regulated reaches / transfer learning). Prefer peer-reviewed DOIs; record rejection reasons if out of scope |
| **Air2stream (Toffolon & Piccolroaz lineage)** | Core 2015/2016 refs present; manuscript already flags unofficial a4/a8 style reference | (i) Confirm official Air2stream citation chain in Related Work; (ii) state that Route-A optional style fit ≠ official comparison; (iii) Route-B TODO: official implementation + documented calibration search before any competitive claim |

### 2.2 Suggested Related-Work paragraph stubs (no results)

1. **Review bridge (Corona 2025):**  
   `[TODO cite corona2025ml]` — locate ThermoRoute among ML stream-temperature
   practices; emphasize issue-time boundaries and clustered inference rather than
   leaderboard RMSE alone.

2. **China-basin transfer / regional ML:**  
   `[TODO add China-basin citations 2021–2026]` — note geographic scope limits of
   the USGS availability-enriched cohort; no national or China-transfer claim on
   Route A.

3. **Hybrid air–water / Air2stream:**  
   `[TODO strengthen Air2stream positioning]` — process-hybrid baseline family;
   unofficial exploratory style reference remains non-claim; official comparison
   deferred.

### 2.3 Additional 2021–2026 scan checklist (optional but useful)

- [ ] Physics-guided / graph river-network temperature (extend beyond Jia 2021, Zwart 2023)
- [ ] Operational or forecast-informed water-temperature ML (to contrast *retrospective* scope)
- [ ] Conformal / uncertainty quantification in hydrology (beyond Romano et al. 2019)
- [ ] Spatial transfer / ungauged baselines (to reinforce that external-30 is *not* ungauged)

Each new entry: Crossref/DataCite verify → `paper/references.bib` → one sentence in §1.1.

---

## 3. Supporting Information (SI) directory skeleton

**Materialized PRE document tree and planned figure inventory** (result files
remain placeholders until evidence exists):

```text
paper/si/
├── README.md                          # SI index + fill rules (mirror §0)
├── SI00_inventory.md                  # checklist of SI items ↔ receipt fields
├── SI01_cohort_and_registry.md        # historical cohort/registry geometry and HUC2 imbalance
├── SI02_information_boundary.md       # issue-time covariates, bridge gate, limitations
├── SI03_model_equations.md            # available identities, units, architecture/identifiability limits
├── SI04_protocol_and_amendments.md    # primary family, inference amendment, errata pointers
├── SI05_comparison_family.md          # five rows, margins, estimands (geometry only)
├── SI06_formal_five_rows_RECEIPT.md
├── SI07_all_model_scores_RECEIPT.md
├── SI08_probability_metrics_RECEIPT.md
├── SI09_development_controls_RECEIPT.md
├── SI10_temporal_coverage_RECEIPT.md
├── SI11_spatial_sensitivity_RECEIPT.md
├── SI12_qc_qualifiers_RECEIPT.md
├── SI13_external_history_arm_RECEIPT.md
├── SI14_missingness_failures_RECEIPT.md
├── SI15_reproduction_hashes_RECEIPT.md
├── SI16_rights_data_dictionary.md
└── figures/
    └── README.md                       # planned FigS1–FigS8 inventory and gates
```

### 3.1 Per-item fill policy

| SI item | May draft now | Numbers |
|---|---|---|
| SI01–SI05 | Yes (design / protocol) | Structural facts from fixed text / registry only |
| SI06–SI14 + FigS3 | Shell captions + empty tables only | Every result cell: `[pending — 探索期数据，不可写入结论]` |
| SI15–SI16 | Schema and acceptance routing only | No replay, rights, DOI or FAIR completion claim without bound receipts/decisions |
| Any “skill vs Air2stream official” table | No until Route B | N/A |

### 3.2 Receipt-shell coverage

The materialized SI documents follow the canonical index in
`docs/POST_PAPER_PROJECTION_DESIGN.md`: formal rows (SI06), all-model scores
(SI07), probability metrics (SI08), development controls (SI09), temporal and
spatial sensitivities (SI10–SI11), QC (SI12), the explicitly history-dependent
external arm (SI13), missingness/failure cases (SI14), reproduction (SI15), and
rights/data-dictionary routing (SI16). Every result/verdict cell uses the
verbatim pending token and cannot be filled from development caches. FigS1–FigS8
remain planned entries in `paper/si/figures/README.md`; no unrendered figure is
described as materialized.

**FigS3:** same five-row geometry as Figure 1d / SI06; every plotted
coordinate remains `[pending — 探索期数据，不可写入结论]` until receipt.

---

## 4. Handoff notes

- This file is the **R2-5 paper skeleton** only; it does not authorize opening,
  alter estimands, or change source hashes.
- Panel (d) and SI06–SI14 must remain empty of scores until a verified receipt
  exists; exploratory Stage-09 caches are not citable.
- SI15–SI16 remain schemas until replay/render receipts and qualified rights
  decisions exist; a visible repository is not a PUBLIC-release authorization.
- Next mechanical steps remain receipt-gated Figure/SI binders and final page QA.
