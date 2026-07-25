# Local output status note

Historical files formerly stored under `outputs/reports`, `outputs/tables`, and
`outputs/figures` came from several superseded experiment generations and have
been removed from the active tree and evidentiary namespace. They remain
recoverable from Git history, which is shipped separately as provenance and still
contains the withdrawn historical blobs, and from a local audit archive outside
the release boundary. They must not be restored as current evidence. Those files included old sample sizes, the already
inspected 2019–2020 development-evaluation split described as a “blind test,”
random station holdouts described as transfer, and ecological/regulatory or
calibration language that the current protocol expressly withdraws.

New files under those directories are authoritative only when they are generated
by the current content-addressed pipeline and satisfy the complete evidence chain
below. A familiar filename alone does not make an artifact current.

The three legacy identifiers `b1`, `s2`, and `p3` are ordinary monitoring sites.
Their file or display order does not establish a reservoir cascade, hydraulic
connectivity, regulation status, or travel time.

Current Route-A evidence is authoritative only when all of the following agree:

1. a content-addressed development run and its validated Stage-09 receipt;
2. the separate matched-control Stage-09b completion receipt;
3. the same-station LSTM Stage-16 exact-closure completion receipt;
4. the pooled external Stage-25 exact-closure completion receipt;
5. the frozen model-suite and isolated development-replay receipts;
6. the pre-label chronology, inference, outcome-QC, and authorization gates; and
7. after the single raw-label acquisition, a fully validated canonical
   `outputs/confirmatory/route_a_*/opening_receipt_v1.json` and its bound trusted
   artifacts.

Absent that complete chain, an output is development-only, incomplete, or
historical. This file is a local convenience note, not an authority document and
is not included in the release. The canonical scope is defined by the root
`README.md`, frozen claim ledger, and byte-verified documents listed there, not by
filenames in this directory.
