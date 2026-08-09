"""Attribute regionalization for the L3 ungauged level (protocol v2).

The L3 anchor is ``clim_hat`` plus a regionalized decay rate ``phi_hat``:
neither uses the target station's own water-temperature record.  This module
implements the pre-registered form (protocol v2, section (a)/(d)):

* a k-nearest-neighbour in attribute z-space (k = 5, Euclidean distance,
  z-scores computed on the in-fold training stations only — never on held
  stations, so the region geometry retains its meaning);
* the response is ``log(phi)`` per station from the official train-fitted
  anchors recorded in the basin-attributes table (the same per-station phi
  whose half-life is analysed in the similarity section);
* stations with missing attributes fall back to the in-fold training median
  of ``log(phi)`` and are counted and logged.

Nothing in this module is fitted on any held-out outcome; the kNN fit happens
per fold over the in-fold training stations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATTRIBUTE_COLUMNS = ("BFI", "log_SR", "log_area", "SNOW_PCT", "FOREST_PCT",
                     "aridity")


class AttributeRegionalizer:
    """kNN regionalizer of log(phi) over static basin attributes."""

    def __init__(self, attributes: pd.DataFrame, k: int = 5):
        self.attributes = attributes
        self.k = int(k)
        missing_cols = [c for c in ATTRIBUTE_COLUMNS if c not in attributes.columns]
        if missing_cols:
            raise ValueError(
                f"basin-attributes table lacks columns {missing_cols}; "
                f"expected {list(ATTRIBUTE_COLUMNS)} plus site_id and phi")

    def regionalize_phi(self, phi: dict[str, float], train_st: list[str],
                        hold_st: list[str]) -> dict[str, float]:
        """Predict ``phi`` for every station; held stations get kNN estimates.

        ``phi`` is the caller's anchor map (pooled values); it is used only
        as a fallback when the attribute table is missing a station.
        """
        attr = self.attributes
        attr = attr.assign(site_id=attr["site_id"].astype(str).str.zfill(8))
        tr = attr[attr.site_id.isin(train_st)].copy()
        if len(tr) < self.k:
            raise ValueError(
                f"fewer training stations with attributes ({len(tr)}) than "
                f"k ({self.k}); L3 is not computable for this fold")
        # z-scores on the in-fold training stations only
        z = (tr[list(ATTRIBUTE_COLUMNS)] - tr[list(ATTRIBUTE_COLUMNS)].mean()) \
            / tr[list(ATTRIBUTE_COLUMNS)].std()
        z = z.fillna(0.0)
        Xtr = z.to_numpy(float)
        ytr = np.log(np.clip(tr["phi"].to_numpy(float), 1e-3, 1.0))

        out: dict[str, float] = {}
        fallback = float(np.exp(np.median(ytr)))
        held_matched = 0
        held_missing = 0
        for site in hold_st:
            row = attr[attr.site_id == site]
            if row.empty:
                out[site] = fallback
                held_missing += 1
                continue
            x = ((row[list(ATTRIBUTE_COLUMNS)].to_numpy(float)[0]
                  - tr[list(ATTRIBUTE_COLUMNS)].mean().to_numpy())
                 / tr[list(ATTRIBUTE_COLUMNS)].std().to_numpy())
            x = np.nan_to_num(x)
            dist = np.linalg.norm(Xtr - x, axis=1)
            knn = np.argsort(dist)[: self.k]
            out[site] = float(np.exp(np.mean(ytr[knn])))
            held_matched += 1
        # training stations keep their own anchors; unknown sites keep the
        # pooled fallback
        for site in train_st:
            out[site] = float(attr.loc[attr.site_id == site, "phi"].iloc[0]) \
                if site in set(attr.site_id) else fallback
        for site, value in phi.items():
            out.setdefault(site, value)
        print(f"regionalize: {held_matched} held stations regionalized, "
              f"{held_missing} fell back to the in-fold median phi "
              f"({fallback:.3f})")
        return out
