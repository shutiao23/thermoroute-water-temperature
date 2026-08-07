# SI02 — issue-time information boundary, replay isolation, and acquisition durability

**Status:** normative specification; no empirical cells.

This document specifies a date-indexed retrospective hindcast boundary, the
isolation model of the pre-acquisition replay, and the transport, publication,
and durability semantics of the test-window acquisition. It is not an operational
replay, an ungauged design, a causal transport model, a security proof, or a
receipt. Any realized availability, score or audit value is
`[pending computation]`.

**What this file is the destination for.** The main text compresses the
information-boundary, replay, and acquisition detail into one or two sentences
each and routes the rest here. Manuscript §3.3 says only that, as a
reproducibility check, a fresh interpreter reloads every trained member and
reproduces the validation, calibration, and 2019--2020 keys and values from the
frozen inputs, and that "the details of that replay are specified in the
Supporting Information". Manuscript §2.3 says only that the test-window daily
means are outcomes only and are not read by model-selection, feature-selection,
threshold-selection, calibration, or station-inclusion code. Sections 3, 4, 5,
and 6 below are the engineering detail behind those sentences. They add no claim
the main text does not make, and they subtract none of its limits.

## 1. Allowed and forbidden information

| Category | PRE rule | Source and role |
|---|---|---|
| issue time | calendar day `t` | `paper/FIGURE1_AND_SI_SKELETON.md` §1a; design symbol |
| target horizons | `h ∈ {1, 3, 7}` days | Figure-1 skeleton §1a; registered geometry |
| WTEMP history | observed target-site history through issue date only | `paper/ThermoRoute_paper.md` §§2.2, 4.1; history-dependent design |
| meteorology | date-indexed Daymet/gridMET values dated no later than historical issue date | PRE manuscript §2.2; retrospective predictor rule |
| frozen references | climatology and damped-anchor inputs | Figure-1 skeleton §1a; model-input role |
| prohibited target | target-date WTEMP | Figure-1 skeleton §1a; leakage prohibition |
| prohibited future predictor | horizon-specific future weather forecast or anticipatory vintage | Figure-1 skeleton §1a; leakage prohibition |

The PRE manuscript records a committed `PASS_EXACT_PRODUCT_BRIDGE` parser/product
compatibility gate for 2018–2020. This SI reports that statement as a frozen
engineering-gate description only; it is neither an test-window receipt nor proof of
as-issued provider availability, matching, operational latency, or local-day
alignment.

## 2. Event-level projection schema

Any later information-boundary audit must bind, rather than infer, the following
fields:

| Field | Required meaning |
|---|---|
| `site_id` | stable target-site identity |
| `issue_date` | date at which the prediction is issued |
| `target_date` | outcome date associated with the horizon |
| `horizon_days` | one registered horizon value |
| `variable_id` | predictor or outcome identifier |
| `source_date` | date represented by a predictor value |
| `vintage_or_retrieval_ref` | bound provider/raw-evidence reference, where applicable |
| `admissibility` | allowed, forbidden, missing, or fail-closed state |
| `source_binding` | manifest path, digest and derivation pointer |

These are SI projection fields. They are not an assertion that current raw
provider bytes, archival vintages, target outcomes, or a completed audit are
available in the repository.

## 3. Replay isolation before any label acquisition

Before label acquisition is authorized, every trained member and every
prediction head is reloaded and re-evaluated by a fresh interpreter started
under `python -I -B` — isolated mode, so no user site directory, no
`PYTHONPATH`, and no `.pth` startup hook can inject code, and no bytecode cache
is written or read. Four capabilities are denied for the lifetime of that
process:

| Denied capability | What it prevents |
|---|---|
| network access | a replay that silently re-fetches instead of recomputing |
| **child processes** | delegation of any step to an unaudited process outside the isolated interpreter |
| repository writes | a replay that repairs the artifact it is supposed to be checking |
| reads from the evaluation namespaces | any path by which a post-2020 outcome could reach the replay |

Inside that boundary the replay reproduces the validation, calibration, and
2019–2020 development keys and values from the frozen inputs. It is a
recomputation, not a comparison of stored summaries. Specifically, it:

1. maps historical panel aliases through the stable station registry, so an
   alias rename cannot silently change which series is being checked;
2. requires every selected station/target-date truth value to equal the
   frozen-panel value;
3. independently recomputes the station-level q90 event thresholds and the
   seasonal references rather than reading them back; and
4. refits the conformal offsets and the Platt calibrators from the exact 2018
   member-averaged rows.

A prediction file that is merely internally self-consistent is rejected, and so
is a re-hashed calibration object. This is what stops a rewritten artifact from
substituting invented truth values or invented calibration parameters for the
real ones.

## 4. Acquisition transport, publication, and durability

There is one logical opening and one fixed request ledger. **This is not a claim
of exactly-once HTTP delivery, and the design does not pretend otherwise.**
Transport may retry, and a response received before its transaction directory is
durable may be requested again. What the transaction discipline does guarantee
is narrower, and is stated as four properties:

| Property | Statement |
|---|---|
| no replacement | a complete, durable, verifiable canonical response is never replaced |
| fail-closed partials | a partial, invalid, or non-canonical transaction fails closed without overwriting anything |
| bounded cleanup | cleanup is confined to unpublished owner-private state; nothing published is removed by it |
| terminal manifest | once the acquisition manifest exists, raw network continuation is permanently disabled and only network-free deterministic recomputation may proceed |

Publication is a single atomic step, not a sequence a crash can interleave.
Normalized tables and the acquisition manifest are generated and validated in a
private staging area on the same filesystem and are published by one directory
rename. The trusted scoring layer is published the same way. A reader therefore
never observes a half-written table: either the rename has happened or it has
not.

## 5. Adversary model, stated as a limit rather than a property

The guards of §4 are **honest-owner crash and replay guards**. They are not a
security property, and this file does not imply one:

- they are **not** protection against a malicious owner;
- they are **not** protection against a same-privilege (same-UID) adversary,
  who can write every path the design relies on;
- the seals, ledgers, and chronology records are repository-internal
  attestations backed by local files and version-control history. They have no
  independent custodian, no external timestamp, and no write-once medium, so
  they must not be represented as independent evidence that no person accessed
  the target outcomes.

The same boundary is recorded in
`protocols/route_a_native_artifact_publication_notice_v1.md`, which states the
evidence boundary in the same terms, and in manuscript §6.4, which describes the
data archive as planned pending a rights review and notes that the self-contained
history bundle retains reachable provenance objects rather than being a
byte-level purge.

## 6. Evidence bindings, and exactly what they bind

The held-out 2021--2023 metric cells of manuscript §4.6 are filled from the
long-form table `outputs/conventional/holdout_metrics_2021_2023.csv` rather than
typed by hand. The scope statements in manuscript §6.1 are plain prose limits,
not machine-delimited claim blocks; they are edited normally.

**Evidence bindings.** Every result slot is filled from an artifact bound by an
exact `{path, sha256}` pair; the renderer re-hashes each file against its binding
before parsing it, and records the artifact path, the artifact digest, and the
field pointer for every rendered cell. A number typed into a slot by hand has no
such lineage and is rejected on that basis, not on the basis of its value.

This is a lineage check, not a security control, and it shares the boundary of
§5.

## 7. Required rendering language

Every information-boundary figure/table must state all of the following:

1. “date-indexed retrospective hindcast”;
2. no horizon-specific future forecasts and no target outcomes as predictor
   inputs; and
3. not operational as-issued availability, not ungauged prediction, and not
   causal transport inference.

Every figure, table, or paragraph that describes §4 or §5 must additionally
state that the guards are honest-owner crash and replay guards and are not
protection against a malicious owner or a same-privilege adversary. Describing
them without that clause is a misstatement of the design.

If a future audit needs an unavailable field, it must show
`[pending computation]` or fail closed; it must not substitute a
development cache or reconstructed timestamp.
