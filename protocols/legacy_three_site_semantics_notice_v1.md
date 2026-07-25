# Legacy three-site semantics notice (v1)

Status: **AUTHORITATIVE CORRECTION; SUPERSEDED BLOBS WITHDRAWN FROM CURRENT
EVIDENCE**

This notice applies only to the legacy identifiers `b1`, `s2`, and `p3` and to
historical repository objects that discussed them.

## Correct interpretation

- `b1`, `s2`, and `p3` are ordinary monitoring stations, not reservoirs.
  No verified metadata establish any upstream/downstream ordering, hydraulic
  connectivity, regulation status, or travel time among b1, s2, and p3. Their
  identifiers, file order, display order, contemporaneous correlations,
  and water-temperature or flow histories do not supply such metadata.
- The repository has no verified station metadata that would support a river
  graph or a physical propagation claim for these three sites.
- The legacy three-site files are not current Route-A evidence. Their source,
  station metadata, measurement dictionary, and redistribution authorization
  remain undocumented.

## Historical correction

Reachable history includes superseded prose, figures, and generated reports in
commits `f0a09ba44296a896205e219984b95e8bfe728219` and
`7c034e8976042899cbe2cc2a2f04322488010d84`. Those objects incorrectly described
the three sites as a regulated multi-reservoir system and interpreted statistical
lag or correlation patterns as a directed physical connection and propagation
time. None of those interpretations is supported by the available metadata.

Commit `08536ce771529b189c4a63a069168b1d0a78c490` introduced the semantic
correction. Later hardening removed the superseded generated outputs from the
active evidence namespace and added executable regression checks.

The old objects remain reachable only so repository chronology can be audited.
They are withdrawn historical provenance, not results, and must not be restored,
cited, summarized, or used to motivate a physical topology. Current documents,
code, and figures must follow the interpretation in this notice.
