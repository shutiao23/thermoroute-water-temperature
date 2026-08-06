# SI07 — all-model exact-common-key scores

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

Rows are generated only from receipt-bound predictions after the renderer proves
one exact common key set for every compared model. Development scores are not
admissible.

| Model | Horizon | paired keys | stations | RMSE (°C) | MAE (°C) | bias (°C) | station-balanced summary | binder row ID |
|---|---:|---|---|---|---|---|---|---|
| *(receipt model)* | *(1/3/7)* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Every cell follows the README cell-level binder contract. The row must bind the
key-registry digest, metric formula, finite-value filter,
station aggregation, unit, rounding rule and parent prediction digest. Selective
model omission rejects the POST build.

The frozen model registry for the temporal cohort is the six primary models
**plus** the seven one-factor architecture controls, so this table carries a row
for each of the thirteen, on one exact common key set, with the controls
labelled exploratory. Figure S10 is the graphical reading of the control rows;
it must agree with this table in value, unit, and rounding.

## Models named in the manuscript that carry no score, and why

Two model rows are absent from every score in the manuscript. Neither absence is
a rendering failure, and neither may be closed by substituting a development
cache, an estimate, or an unofficial reimplementation.

### 1. The air2stream-style hybrid reference — `NOT_RUN`

**Status.** The headline development table records
`Air2stream-style a4/a8 (unofficial, non-primary): NOT_RUN; headline entry is
NOT_RUN / NA`. No score exists at any lead, in any arm, for any period. The
manuscript removed it from the Abstract, the Key Points, the Introduction, the
reference-set enumeration, the score table, the Conclusions, and the cover
letter, and states in §3.2 that it is excluded from every claim in the paper and
in §6.2 that the hybrid process family is consequently absent from every
comparison. This file is where its status lives.

**Provenance.** What exists is a *style* reference, not the published model. The
pinned-source audit (`docs/AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md`) records
verdict `BLOCKED_NO_COMPILER / REFERENCE_CASE_NOT_ATTESTABLE` against upstream
commit `d4834bccf01657c03ab60efb4c18f8a256132c53`:

| Finding | Consequence for a comparison |
|---|---|
| no supported Fortran compiler present (`gfortran`, `ifort`, `ifx`, `flang`, `nvfortran`, `f95`, `f90` all absent), no Makefile or documented compiler command, and Intel-specific `ifport` / `makedirqq` calls in `AIR2STREAM_READ.f90` | the pinned source was never built in this environment |
| the shipped `air2stream_1.0.0.out` binary was deliberately not executed | running an upstream binary would not show that the pinned source builds, and no golden-output checksum is published to attest it against |
| the commit is unsigned (`%G? = N`, `git verify-commit` fails) and no tag is advertised | provenance is TLS transport plus Git object identity, not signer attestation |
| no committed expected output; the default PSO path calls `random_seed`/`random_number` with no fixed-seed contract | any reference case must use tolerance-based declared metrics or an explicitly frozen deterministic seed path, never byte-identical optimizer output |
| README attributes preprocessed case inputs to FOEN and MeteoSwiss; the CC BY-SA 3.0 code licence does not by itself resolve case-data redistribution | the reference case cannot be redistributed with this work as it stands |

**What a defensible comparison would require**, all of it before any number is
quoted:

1. the official implementation built from the pinned source in a recorded
   environment, with the build command and toolchain versions in the receipt;
2. an attested reference case — either a frozen deterministic seed path or
   declared metrics with a stated tolerance — reproduced from that build;
3. a documented, symmetric calibration search: the same predeclared budget,
   objective, and selection partition the learned references receive, since
   tuning budgets are already an acknowledged asymmetry in this study;
4. scoring on the identical exact common key registry every other model uses, so
   the row shares this table's denominators; and
5. a resolved redistribution decision for the case data, recorded in SI16.

Until all five hold, the correct rendering is the status token, not a number and
not a blank.

### 2. Per-station LightGBM — named, unscored

The reference set names a per-station LightGBM variant, whose purpose is to
isolate the value of pooling against the global model with site identity as a
categorical feature. **No scored development summary of that variant exists**,
so the manuscript names the variant in §3.2 without quoting a number and omits it
from the §4.1 score table. It is not a primary model and appears in neither the
five-row comparison family nor the protocol's model registry, so its absence
weakens no registered claim. Closing this gap means scoring it on the same exact
common key registry and adding a row here; it does not mean reporting the
global-model number under the per-station label.
