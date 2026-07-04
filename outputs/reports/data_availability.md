# Open Research / Data & Software Availability Statement (AGU format)

> Paste this into the manuscript's **Open Research** section and replace the DOI
> placeholder after the Zenodo upload. AGU/WRR require a persistent identifier
> (DOI) in a trusted repository — a GitHub-only link is **not** compliant.

**Software.** The ThermoRoute code, configuration, input station panels, and
derived artifacts used to produce every figure and table in this study are
archived at Zenodo (https://doi.org/REPLACE_WITH_ZENODO_DOI, version 1.0.0;
MIT license) and developed at
https://github.com/REPLACE_ORG/thermoroute-water-temperature. The archive
contains a SHA-256 manifest (`scripts/14_manifest.py`) certifying every derived
artifact, a locked dependency set (`requirements-lock.txt`), and a continuous-
integration workflow that reproduces the reported metrics within documented
tolerances. Predictions and model checkpoints (multi-hundred-MB) are regenerable
from the archived code and panels via `scripts/run_all.sh`.

**Data.** The analysis is built from three openly available sources, redistributed
in the archived station panels in derived form:

- **Water temperature and discharge** — U.S. Geological Survey National Water
  Information System (USGS NWIS), retrieved with the `dataRetrieval` R package
  (De Cicco et al., 2024); https://waterdata.usgs.gov/nwis. USGS data are public
  domain.
- **Air temperature and precipitation** — Daymet Daily Surface Weather (ORNL
  DAAC); https://doi.org/10.3334/ORNLDAAC/2129.
- **Wind speed** — gridMET (Abatzoglou, 2013); https://www.climatologylab.org/gridmet.html.

No new observational data were collected. Station identifiers (USGS site numbers)
for all 120 gages are listed in `outputs/tables/usgs_stations_with_huc.csv` within
the archive, enabling exact re-retrieval of the raw series.
