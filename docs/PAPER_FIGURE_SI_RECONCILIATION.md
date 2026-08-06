# Figure and SI reconciliation after the benchmark restructure — 2026-08-06

| Field | Value |
|---|---|
| Closes | the three gaps `docs/PAPER_BENCHMARK_RESTRUCTURE.md` §10 left open: the figure-specification conflict, the stale LaTeX, and the SI sections that must receive relocated material |
| Branch | `feat/route-a-completion` |
| Files changed | `paper/FIGURE_REDRAW_SPEC.md`, `paper/agu_submission/figures/render_post_main_figures_skeleton.py`, `paper/si/figures/render_post_supporting_figures_skeleton.py`, `paper/ThermoRoute_paper.md`, `paper/agu_submission/build_agu.py`, `paper/agu_submission/README.md`, `paper/agu_submission/ThermoRoute_WRR.{tex,pdf}`, `paper/si/SI00,SI02,SI07,SI08,SI11,SI16`, `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md`, this file |
| Files deliberately NOT touched | `src/`, `scripts/`, `tests/`, `protocols/`, `pyproject.toml`, `requirements*.txt`, `.github/`, `ops/`, `outputs/`; both sibling worktrees (read-only, except the two-file round trip in §5) |
| Committed | nothing |
| Numbers invented | none. Every value below is transcribed from a named artifact |

---

## 1. The conflict, stated precisely

The restructure needs four main figures: (1) station map and cohort geometry;
(2) how baseline choice changes reported skill; (3) how random held-site versus
whole-region holdout changes the transfer conclusion; (4) regional and seasonal
heterogeneity with conformal coverage–width.

`paper/FIGURE_REDRAW_SPEC.md` specified four different figures, and the
restructure's own §7 named three mismatches: the new Figure 3 had no main-figure
slot, the new Figure 4 merged the old Figure 3(a), 4(c) and S7, and the old
Figure 4's ablation content had to demote to SI. It also proposed two panels that
the spec's own governing rule forbids.

Resolution principle, applied throughout: **no panel is deleted.** Every panel of
the previous four main figures either keeps a main-text slot under a new number
or moves to a named SI figure, and the two genuinely new main panels are built
from evidence that already exists.

## 2. The figure delta, panel by panel

| Was | Content | Is now | Evidence period | Why |
|---|---|---|---|---|
| Fig 1(a) | persistence challenge | **Fig 1(c)** | PRE structural | unchanged content, reordered |
| Fig 1(b) | bounded-correction schematic | **Fig S3(c)** | PRE structural | the architecture is now the object under test, not the contribution; a thesis-level mechanism panel no longer belongs in the opening figure, and S3 already owns the full bound/calibration dataflow |
| Fig 1(c) | cluster geometry against the gate | **Fig 1(b)**, extended with the HUC2/4/6/8 ladder | PRE structural | the ladder is what makes the "threshold designed to fail" reading answerable inside the figure |
| — | station map + nearest-neighbour scale | **Fig 1(a)** (new) | PRE structural | the restructure needs it; the full registry geometry stays in Fig S1, so S1 still expands rather than duplicates |
| — | reference ladder | **Fig 2(a)** (new) | target | built at target period from the trusted scorer's own reference models |
| Fig 2(a)–(c) | all-model station distributions | **Fig 2(b)–(d)** | target | shifted by the new panel (a) |
| Fig 2(d) | registered five-row forest | **Fig 2(e)** | target | shifted |
| Fig 3(a) | coverage–width plane | **Fig 4(c)** | target | the restructure keeps only the aggregate plane in the main text |
| Fig 3(b) | event score | **Fig S5(b)** | target | S5 already carried the score matrix; the Brier-skill panel joins it |
| Fig 3(c)–(e) | reliability by horizon | **Fig S5(c)** | target | S5 already carried every registered bin |
| — | three transfer arms, fold geometry, distance association, held-region ranking | **Fig 3(a)–(d)** (promoted) | **development 2019–2020** | this is the restructure's missing main figure; see §2.2 |
| Fig 4(a) | seven registered architecture interventions | **Fig S10(a)–(b)** (new SI figure) | target | demoted per the restructure; see §2.3 |
| Fig 4(b) | attrition waterfall | **Fig S6(a)** | target | S6 already used the same denominator-preserving construction; it gains the reportable-cluster stage |
| Fig 4(c) | eight temporal-coverage candidates | **Fig 4(b)** | target | stays in the main text |
| Fig 4(d) | external history-dependent arm | **Fig S8(b)** | target | S8(b) already owned that cohort's scope statement; it gains the results |
| — | per-HUC2 regional heterogeneity, aggregated | **Fig 4(a)** | target | every unit and every leave-one omission stays in Fig S7, which now expands Fig 4(a) |

### 2.1 Two proposals refused, and where their content went

`paper/FIGURE_REDRAW_SPEC.md` §1 states: **one figure never mixes two evidence
periods.** The restructure brief's §7 proposed two panels that violate it.

| Refused | Why | Content routed to |
|---|---|---|
| Fig 2(a) as a *development-period* panel with a scope band, inside an otherwise target-period figure | a reader comparing a ladder rung with a station in the next panel would be comparing 2019–2020 with 2021–2023 | the ladder is rebuilt at target period over the five reference models the trusted scorer emits; the development-period ladder (+0.251 against persistence versus +0.038 against damped persistence at 7 d) is Figure 3(a), row 1 |
| Fig 4(d), "what calibration costs", as a *development-period* panel inside an otherwise target-period figure | same hazard, on a coverage–width plane where the mixing would be invisible | Figure S9 panels (a)–(b), which already carry the split-CQR, block-maximum and delayed-ACI contrast with `EvidencePeriod.DEVELOPMENT` and a mandatory scope band |

Both refusals are recorded in the skeletons as `DroppedPanel` entries with their
reason and their substitute, so `--status` prints them and they cannot be
silently re-proposed. The rule is now **machine-enforced** rather than editorial:
`PanelSpec.evidence_period` may be left empty (inherit) but may never disagree
with its figure, and `NAMESPACE_EVIDENCE_PERIOD` extends the same rule to
`shares_value_ids_with` — a target-period figure can no longer declare that it
shares value IDs with a development-period namespace. Both raise `ManifestError`
before any gate is read.

### 2.2 Why the new Figure 3 is development-period, and still POST-gated

The one-time opening produces **no held-region artifact**. The confirmatory
protocol registers a temporal cohort and a site-identifier-disjoint external
cohort; a leave-one-HUC2-region-out arm is in neither, and
`docs/R13_POSTOPEN_TABLE_RENDERER.md` §6 records that Table 4.6's held-region
fragment renders as `NOT_EMITTED_BY_THE_ONE_TIME_OPENING`. The whole-region
holdout evidence in this paper is the Stage-13b/13c development-period evidence,
permanently. Running a target-period regional holdout would be a protocol
amendment, not a figure change.

Figure 3 therefore declares `EvidencePeriod.DEVELOPMENT`, renders a mandatory
in-panel scope band, and forbids numerical comparison with any target-period
figure. It nonetheless keeps the verified opening receipt as a **required**
dependency, like every other POST figure. That gate does not supply its numbers;
it proves the submission is past the one-shot boundary, so a development display
cannot be published as a stand-in for a target-period result nobody attempted.

Its evidence resolves today — `outputs/tables/region_transfer.csv`,
`outputs/reports/region_transfer.md`, `outputs/reports/tuurt.md`,
`data_usgs/station_registry_v1.csv` all read `yes` under the multicore evidence
root — so like Figure S9 it is blocked only by the spec state and the receipt.

### 2.3 Why the demoted ablation content became a target-period SI figure

The seven one-factor architecture controls are **not** development-only. The
confirmatory protocol's `mandatory_exploratory_architecture_controls` list is
resolved into the temporal cohort's required model set beside the six primary
models (`opening._required_models`), so the trusted scorer emits a row for every
control on the same exact common keys, and manuscript Table 4.2 transcribes them.
Demoting them to SI is a change of prominence, not of evidence class, so Figure
S10 is target-period and keeps the old Figure 4(a) gate verbatim: all seven
control rows under the final model/seed contract, or the figure is not generated
and no development control is substituted.

The Stage-09b information-matched controls (plain causal TCN, plain MLP) *are*
development-period. They stay a non-blocking qualifier on Figure S10 and a table
in SI09, and Figure S10 declares no shared value ID with the `si09` namespace —
which `validate_manifest` would now refuse as a cross-period share.

### 2.4 Final manifest

Fourteen figures: four main, ten supporting.

| Fig | Stem | Panels | State | Period | Anchor |
|---|---|---:|---|---|---|
| 1 | `fig01_preopening_concept` | 3 | `PRE_MATERIALIZED_REDRAW_STALE` | PRE structural | F1, §1 close |
| 2 | `fig02_reference_ladder_point_performance` | 5 | `POST_TEMPLATE_ONLY` | target | F2, §4.1 close |
| 3 | `fig03_spatial_partition_transfer` | 4 | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **development** | F3, §4.4 close |
| 4 | `fig04_heterogeneity_and_interval_cost` | 3 | `POST_TEMPLATE_ONLY` | target | F4, §4.5 close |
| S1 | `figS1_cohort_registry` | 4 | `PRE_MATERIALIZED` | PRE structural | S1, §2.1 |
| S2 | `figS2_information_boundary` | 4 | `PRE_MATERIALIZED` | PRE structural | S2, §2.3 |
| S3 | `figS3_model_architecture` | 4 | `PRE_MATERIALIZED_DESIGN` | PRE structural | S3, §3.1 |
| S4 | `figS4_point_heterogeneity` | 3 | `POST_TEMPLATE_ONLY` | target | S4, §4.2 close |
| S5 | `figS5_probability_diagnostics` | 4 | `POST_TEMPLATE_ONLY` | target | S5, §4.6 |
| S6 | `figS6_temporal_attrition` | 4 | `POST_TEMPLATE_ONLY` | target | S6, §4.6 |
| S7 | `figS7_spatial_leave_huc2` | 4 | `POST_TEMPLATE_ONLY` | target | S7, §4.5 close |
| S8 | `figS8_qc_external_failures` | 3 | `POST_TEMPLATE_ONLY` | target | S8, §4.6 |
| S9 *(optional)* | `figS9_conformal_sensitivity_development` | 4 | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **development** | S9, §6.3 |
| S10 | `figS10_architecture_interventions` | 2 | `POST_TEMPLATE_ONLY` | target | S10, §4.6 |

The manuscript's `FIGURE_ANCHOR` set is exactly `{F1..F4, S1..S10}` — fourteen
ids, verified by regex — with `state=POST_DEVELOPMENT` on F3 and S9 and `PRE` on
F1 and S1–S3. Two anchors moved: S4 from the §4.3 close to the §4.2 close (S4
expands Figure 2's point evidence, and §4.3 is development-period prose whose
controls are tabulated in SI09), and S10 is new at §4.6.

### 2.5 Verification

```
$ python -m py_compile paper/agu_submission/figures/render_post_main_figures_skeleton.py \
                       paper/si/figures/render_post_supporting_figures_skeleton.py
COMPILE OK
```

Both `--status` runs exit **2** against the multicore evidence root, writing no
file and creating no axes. Every figure is blocked on `spec_state` and
`opening_receipt`; the eleven POST figures are additionally blocked on the
`trusted/` artifacts the opening will create. Qualifier rows never block. Both
refusals of §2.1 appear as `dropped_panel` qualifier rows. `figS9`'s
`evidence_period` line is byte-identical to its previous output; `fig03`'s reads
`development_2019_2020 (Stage-13c development-evaluation span
2019-01-01..2020-12-31; confirmatory target starts 2021-01-01); in-panel scope
band fig03.scope.development_period_not_confirmation is mandatory`.

Preserved unchanged: the fail-closed double gate, exit 2 with no file and no
axes, `--status` as the only read-only mode, the `PanelBuilderNotImplemented`
and `PostGateNotPassed` catch, the `TEMPLATE_ONLY` substring test, the
`[A-Z0-9_]+` state regex, `ValueBinder.mark`/`require_contract_ids`/
`require_scope_band`/`cross_check_shared_value_ids`, the `evidence_period` field,
the in-panel scope band, and the one-period-per-figure rule — now enforced at two
further levels.

---

## 3. LaTeX regeneration

`paper/agu_submission/ThermoRoute_WRR.tex` was regenerated from the restructured
Markdown with `paper/agu_submission/build_agu.py` and compiled with `latexmk`
against the official `agujournal2025` class.

| | Previous build (HEAD `.tex`) | This build |
|---|---:|---:|
| Pages | 44 | **43** |
| Errors | 0 | **0** |
| Overfull boxes | 0 | **0** |
| Missing characters | 0 | **0** |
| Undefined references / citations | 0 | **0** |
| Underfull `\hbox` | 979 | 937 |
| Underfull `\vbox` | 41 | 36 |
| `LaTeX Font Warning` | 5 | 6 |
| `Package hyperref Warning` | 2 | 2 |

The baseline column was produced by compiling `git show
HEAD:paper/agu_submission/ThermoRoute_WRR.tex` in a scratch directory with the
same class assets, so the two columns are comparable.

**Warning classes, all of them.** (a) Underfull `\hbox`/`\vbox` — the cost of the
`\sloppy` the generator sets deliberately (its comment records that removing it
reintroduces four overfull boxes); both counts fell. (b) Six `LaTeX Font
Warning`s: two 10.5 pt size substitutions to 10.95 pt (≤ 0.45 pt, pre-existing),
`OMS/cmtt/m/n` undefined substituted by `OMS/cmsy/m/n` (pre-existing),
`OML/cmtt/m/n` undefined substituted by `OML/cmm/m/it` (**new**), and the two
summary lines. The new one comes from the Greek `\ensuremath{}` mappings now
needed inside code spans; it is a shape substitution to the standard math font,
not a missing glyph, and `Missing character` is 0. (c) Two `Package hyperref
Warning`s: "Draft mode on" and "Height of page (\paperheight) is invalid" — both
are consequences of `\documentclass[draft]`, which is the submission branch this
package targets, and both were present before.

### 3.1 What tripped, and what was changed

**The required-content check tripped.** `build_agu.REQUIRED_STATUS_TEXT` demands
the literal `clearly marked slots`; the restructured Manuscript-status section
said "marks the evaluation-period tables as slots". **Manuscript reworded**, not
the check: "…and leaves the evaluation-period tables as clearly marked slots".

**The Key Points check passed** — 128, 127, 126 characters against the
140-character limit. Nothing was changed for it.

**The undeclared-Unicode check tripped on 23 codepoints**, all introduced by the
restructure's equations (1)–(10). This check offers two remedies in its own error
message; both were used, each where it is honest:

* *Mapping added* for 21 codepoints — Δ Σ β δ κ λ π ρ σ τ φ ‖ … ⁴ ₁ ℓ ∈ √ ≈ ⟨ ⟩.
  `UNICODE_DECLARATIONS` exists for exactly this, and its own comment says the
  entries are cheap and keep prose edits from breaking the build.
* *Manuscript reworded* for the two combining marks, U+0302 and U+0303. A
  `\DeclareUnicodeCharacter` mapping receives a combining mark **after** its base
  letter while LaTeX accent commands are prefixes, so no honest mapping exists —
  any mapping would drop or misplace the accent. `q̃` (standardized
  log-discharge) became `q`, and `q̂_τ` (the quantile head in equation 8) became
  `Q_τ`; both are defined in place, and `Q` collides with nothing.

**One permanent-constraint lint fired** and is covered in §5.

### 3.2 The PRE-OPEN render guard

`build_agu.main()` calls `assert_preopen_manuscript_render_allowed`, which has
two arms: a **phase** arm (refuse once an opening authorization or a confirmatory
namespace exists) and a **freeze** arm (three manuscript SHA-256 values pinned in
`protocols/route_a_claim_registry_v1.json`). The freeze arm has been failing by
design since the 2026-08-05 restructure rewrote all three files; re-sealing it is
a `protocols/` edit, which is in the gating source tier and is separately
authorized (`docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §0.2; the escape is
R6, already queued in the remediation lineage).

The regeneration therefore re-asserted the **phase** arm explicitly — neither
`data_usgs/confirmatory_opening_authorization_v1.json` nor
`outputs/confirmatory/` exists — and then called `build_agu._render()` directly.
Every content check ran unchanged: eight claim blocks, the manuscript-status
text, fifteen result-slot markers *and their survival through Pandoc conversion*,
three Key Points under 140 characters, the banned phrase "quantile crossing",
the eleven withdrawn-claim patterns, the legacy-semantics scanner, and the
undeclared-Unicode check. **No check was loosened.** `build_agu.py --check` still
refuses in this worktree, which is the honest state.

---

## 4. SI relocation map

Each destination was checked for three things: that it actually contains the
relocated material, that it reads as a standalone section, and that it does not
contradict the main text.

| Destination | Received | Main-text sentence it now backs | Status |
|---|---|---|---|
| **SI02** | replay isolation model (`python -I -B`; the four denied capabilities including child processes; the four-step recomputation list); acquisition transport and durability (not exactly-once HTTP; no-replacement, fail-closed-partials, bounded-cleanup, terminal-manifest; staging plus single directory rename); adversary model; body-hash semantics | §3.3 "the isolation model, the byte bindings, and the publication and crash semantics of that replay are specified in the Supporting Information"; §3.7 "honest-owner crash and replay guards rather than protection against a malicious owner or a same-privilege adversary; those details are in the Supporting Information"; §4.6 "rejected by body hashes and evidence bindings" | **filled**, retitled, renumbered §1–§7; §7 now requires the honest-owner clause in any rendering of §4/§5 |
| **SI07** | air2stream `NOT_RUN` status, the five findings of the pinned-source build audit, and the five conditions a defensible comparison would require; the unscored per-station LightGBM variant; the thirteen-row temporal registry | §3.2 "its status, provenance, and what a defensible comparison would require are reported in the Supporting Information"; §6.2 "the hybrid process family is absent from every comparison"; §4.6 Table 4.2 control rows | **filled** |
| **SI08** | already carried the Stage-19 disposition and the measured degeneracy table | §6.3 | **verified**; added §4, the figure-routing table, so SI08 and the manifest cannot drift; corrected the two "Figures 3 and S5" pointers to "Figure 4 and Figure S5" |
| **SI11** | the recomputed HUC2/HUC4/HUC6/HUC8 ladder with effective counts, fractions and largest shares; why HUC8 was not adopted; the leading-zero recomputation trap; the figure relationship | §3.6 "recomputed with the gate's own code on the frozen registry…"; Supporting Information paragraph "including the recomputed cluster geometry at HUC2, HUC4, HUC6, and HUC8" | **filled** |
| **SI16** | the four redistribution classes defaulting to exclusion, the two excluded outright with their rights basis, the substitute-for-the-bytes rule, the exact scope of the MIT licence, and the irreversible 2006–2020 provider-byte gap | §8 "The classes currently defaulting to exclusion, and the two excluded outright … are enumerated in SI16" | **filled** |
| **SI00** | reconciled with the fourteen-figure manifest and the SI file set; five inventory rows rewritten; source map updated | — | **reconciled** |

No destination contradicts the main text. Three specific consistency points were
checked by hand:

1. SI08 says the *development-period* probabilistic stage is not produced, and
   that everything it projects is *target-period* from the trusted scorer. The
   manuscript §6.3 says the same and explicitly routes the evaluation-period
   family to Table 4.4. No contradiction.
2. SI02 §5 states the honest-owner boundary in the same terms as manuscript §3.7
   and as `protocols/route_a_native_artifact_publication_notice_v1.md`. No SI
   sentence upgrades a guard into a security property.
3. SI11 states that HUC8 clears the cluster gate and returns the same verdict,
   matching manuscript §3.6 exactly, and never presents it as a route to
   eligibility.

---

## 5. Validator outcome

```
cd /home/lzq/workspace/parttime/thermoroute-remediation && PYTHONPATH=src \
  /home/lzq/anaconda3/envs/route-a/bin/python scripts/26_validate_claims.py \
  --root . --registry protocols/route_a_claim_registry_v1.json
Route-A claims OK
EXIT=0
```

Only `paper/ThermoRoute_paper.md` and `paper/agu_submission/ThermoRoute_WRR.tex`
were copied into that worktree; both were copied back afterwards and verified
byte-identical by SHA-256 (`b3a5cf2f…` and `de84e612…`). No `git stash`,
`checkout`, `restore`, `reset`, or `clean` was used anywhere.

### 5.1 The one lint that fired, and the rewording

First run:

```
- LINT P03_NOT_OPERATIONAL_REPLAY: paper/ThermoRoute_paper.md:236:
    operational(ly)? (ready|validated|replay|forecast skill)
- LINT P03_NOT_OPERATIONAL_REPLAY: paper/agu_submission/ThermoRoute_WRR.tex:309: (same)
```

§2.3 read "This is a date-indexed retrospective hindcast, not an operational
replay: …". The guarded bigram may appear only inside the sanctioned claim
blocks, and the regex does not read negation, so the free-text occurrence fires
even though the sentence denies the property. The single TeX hit is the same
sentence after conversion.

Reworded to "This is a date-indexed retrospective hindcast, and not a
re-execution of what a forecaster could have run on the day (Section 6.1): …".
The meaning is unchanged, the cross-reference points the reader at the sanctioned
claim block, and `LIMIT_NOT_OPERATIONAL_REPLAY` in §6.1 still carries the exact
phrase verbatim with its body hash intact. The `.tex` and PDF were rebuilt after
the rewording; the page count and every warning count above are from that final
build.

---

## 6. Blocked, and on whom

| # | Item | Owner | Note |
|---|---|---|---|
| 1 | **Figure 1's committed bytes are stale.** Panel (b) changed from the bounded-correction schematic to the station map, so `fig01_preopening_concept.{svg,pdf,png,json,csv}` no longer match the spec, and both PRE manifests bind a spec SHA-256 that this revision changes | figure track | the PRE renderer must be re-run; this task did not render figures. State token set to `PRE_MATERIALIZED_REDRAW_STALE` so the mismatch cannot be missed |
| 2 | **All eleven POST figures are blocked on the one-time opening**, as designed | opening owner | `--status` exit 2 for every one |
| 3 | **Pre-opening degeneracy guard has not been run against the target period** | opening owner | `opening.py:7093-7094` can abort the one-shot opening on a single zero-width member-averaged interval. Development-panel measurement is clean for every model in the confirmatory registry; the target period is untested. Carried as a provenance qualifier on Figure 4 and Figure S5 |
| 4 | **`preopen_document_sha256` is stale for all three manuscript sources and for the regenerated `.tex`** | protocol/lineage owner | a `protocols/` edit in the gating tier; the escape is R6, already queued in the remediation lineage. Until then `build_agu.py --check` refuses in this worktree |
| 5 | **Per-station LightGBM has no scored summary** | analysis owner | recorded in SI07; §3.2 names the variant without a number and §4.1 omits it. Not a primary model, so no registered claim is weakened |
| 6 | **air2stream cannot be compared** | external | five conditions listed in SI07, starting with a Fortran toolchain that does not exist in this environment |
| 7 | `[TRAINING WALL-CLOCK …]`, `[INFERENCE COST …]`, author block, DOIs, licence, release tag, repository URL | authors / rights review | unchanged by this task; tracked in `docs/WRR_SUBMISSION_CHECKLIST.md` |
