# Figure 1 and SI skeleton (R2-5)

**Status:** pre-opening design scaffold only.  
**Constraint:** no fabricated performance numbers. Every result cell is
`[pending — 探索期数据，不可写入结论]`. Values may be filled only from a
verified opening / completion receipt after Route-A authorization.

**Canonical manuscript:** `paper/ThermoRoute_paper.md`  
**Do not edit:** `src/`, `scripts/`, `tests/`, `protocols/` (source-hash tree).

---

## 0. Antagonistic fill rules

| Allowed now | Forbidden until receipt |
|---|---|
| Layout, captions, axis labels, equation forms | RMSE, MAE, skill, coverage %, Brier, etc. |
| Frozen design facts already sealed in manuscript text (e.g. ≤15 HUC2, gate thresholds) | Invented effect sizes, CI endpoints, p-values |
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
a_{t+h}
+
\delta\,\tanh\!\Bigl(\frac{r_{t+h}}{\delta}\Bigr),
\qquad
\delta = \texttt{delta\_scale}
\]

where \(a_{t+h}\) is the frozen damped-persistence anchor and \(r_{t+h}\) is the
learned unrestricted residual proposal. With finite \(\delta\), the point forecast
lies in \([a_{t+h}-\delta,\,a_{t+h}+\delta]\).

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
- Callouts already sealed in manuscript design text (**cohort geometry / gate
  inputs only — not RMSE, skill, or any scored performance**):
  - at most **15** reportable HUC2 clusters;
  - stations per HUC2 range **2–26** (pre-attrition design description);
  - largest-cluster share ≈ **21.7%** of stations;
  - inverse-Herfindahl effective cluster count ≈ **9.54**;
  - gate requires \(n_{\mathrm{clusters}}\ge 30\), effective fraction \(\ge 0.75\),
    largest share \(< 0.25\) → **necessarily fails** → permanent
    `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` / fixed-cohort descriptive only.
  - Any redraw-derived count that is *not* yet re-verified from the frozen
    registry must stay labeled `registry-derived (redraw TODO)` and must **not**
    be replaced by exploratory Stage-09 scores.

**Caption draft (EN):**

> (c) Station counts by HUC2 for the availability-enriched frozen cohort.
> With ≤15 clusters, the minimum-30-cluster inference gate fails before any
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
> (`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`); p-values and CIs are
> assumption-conditional sensitivities, not superiority / national claims.

**Caption draft (EN):**

> (d) Predeclared five-comparison effects and cluster-bootstrap intervals.
> Values are withheld pending a verified receipt; the panel geometry is fixed
> so receipt fields can populate without redesign.

---

### Figure 1 production checklist

- [ ] Draft vector layout (Inkscape / matplotlib / TikZ) matching 2×2 grid
- [ ] Panel (c) redraw script from `data_usgs/station_registry_v1.csv` (no scores)
- [ ] Panel (d) binder: map receipt JSON fields → table cells (fail-closed if missing)
- [ ] Bilingual 非安全界 callout locked in art file
- [ ] AGU figure file path TBD under `paper/agu_submission/figures/` (not created yet)

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

**Proposed tree** (files are placeholders; create content only when evidence exists):

```text
paper/si/
├── README.md                          # SI index + fill rules (mirror §0)
├── SI00_inventory.md                  # checklist of SI items ↔ receipt fields
├── SI01_cohort_and_registry.md        # 120-site registry, HUC2 imbalance, missingness
├── SI02_information_boundary.md       # issue-time covariates, bridge gate, limitations
├── SI03_model_equations.md            # full identities: anchor, router, TCN, bounded residual, CQR/Platt
├── SI04_protocol_and_amendments.md    # confirmatory family, inference amendment, errata pointers
├── SI05_comparison_family.md          # five rows, margins, estimands (geometry only)
├── SI06_primary_effects_RECEIPT.md    # ← ALL CELLS: [pending — 探索期数据，不可写入结论]
├── SI07_interval_and_probability_RECEIPT.md
├── SI08_architecture_controls_RECEIPT.md   # Stage 09 / 09b sensitivities
├── SI09_temporal_coverage_audit_RECEIPT.md
├── SI10_external_cohort_RECEIPT.md         # 30-site metadata-disjoint arm
├── SI11_air2stream_style_note.md      # unofficial a4/a8; no official claim
├── SI12_reproducibility_hashes.md     # source_tree_hash, panel digest, chronology pointers
└── figures/
    ├── FigS1_huc2_station_counts.svg      # (c)-style enlarge
    ├── FigS2_information_boundary.svg
    ├── FigS3_effect_forest_RECEIPT.svg    # receipt-derived only
    └── FigS4_missingness_maps.svg
```

### 3.1 Per-item fill policy

| SI item | May draft now | Numbers |
|---|---|---|
| SI01–SI05, SI11–SI12 | Yes (design / protocol) | Structural facts from sealed text / registry only |
| SI06–SI10 + FigS3 | Shell captions + empty tables only | Every cell: `[pending — 探索期数据，不可写入结论]` |
| Any “skill vs Air2stream official” table | No until Route B | N/A |

### 3.2 SI06 table shell (example)

| Comparison | h | Effect °C | CI_lo | CI_hi | raw p | Holm p | eligibility |
|---|---:|---|---|---|---|---|---|
| TR − damped | 1 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| TR − damped | 3 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| TR − damped | 7 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| TR − LGB | 3 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| TR − LGB | 7 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

### 3.3 SI07–SI10 receipt shells (pending markers required)

All numeric / verdict cells below use the verbatim token
`[pending — 探索期数据，不可写入结论]`. Do not fill from exploratory caches.

**SI07 — interval / probability (receipt-derived)**

| Horizon | Coverage % | Interval width | Brier / prob score | eligibility |
|---|---|---|---|---|
| 1 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 3 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 7 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

**SI08 — architecture controls (Stage 09 / 09b; receipt-derived)**

| Control arm | Horizon | Δ vs reference | CI / sensitivity | verdict |
|---|---:|---|---|---|
| *(row per sealed control)* | *h* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

**SI09 — temporal coverage audit (receipt-derived; not a skill claim)**

| Window / stratum | n stations | coverage / completeness | score cells |
|---|---|---|---|
| *(stratum)* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

**SI10 — external 30-site metadata-disjoint arm (receipt-derived)**

| Comparison | h | Effect °C | CI | eligibility |
|---|---:|---|---|---|
| *(external arm row)* | *h* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

**FigS3:** same five-row geometry as Figure 1d / SI06; every plotted
coordinate remains `[pending — 探索期数据，不可写入结论]` until receipt.

---

## 4. Handoff notes

- This file is the **R2-5 paper skeleton** only; it does not authorize opening,
  alter estimands, or change source hashes.
- Panel (d) and SI06–SI10 must remain empty of scores until a verified receipt
  exists; exploratory Stage-09 caches are not citable.
- Next mechanical steps (out of scope here): draw Fig.1 art; expand China-basin
  bib TODOs; optionally materialize empty `paper/si/` stubs.
