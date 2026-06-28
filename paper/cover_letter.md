**Cover letter**

To the Editor-in-Chief
*Journal of Hydrology*

Dear Editor,

We submit our manuscript, **"ThermoRoute: a dynamic thermal-memory prior with
calibrated, transferable multi-station river water-temperature forecasting,"** for
consideration as a research article in *Journal of Hydrology*.

Daily river water temperature is a master variable for aquatic ecology and the
operation of regulated rivers, yet two problems recur in the machine-learning
literature on the subject: persistence is an extraordinarily strong baseline that is
often not properly benchmarked against, and reported skill frequently depends on
covariates that would not be available when a forecast is actually issued. Our work
confronts both directly and, we believe, sets a more honest standard for the
problem.

We make four contributions, each established with statistical rigour on a
40-station, public-domain large sample (USGS gages with Daymet and gridMET forcing),
under a strict historical-information protocol with a one-shot 2019–2020 blind test:

1. **A physics prior that contains the strong baseline.** ThermoRoute's dynamic
   thermal-relaxation prior reduces exactly to damped persistence toward climatology,
   so any gain is attributable to the learned component. On the large sample it
   significantly beats persistence and damped persistence at all lead times
   (per-station Wilcoxon p ≤ 10⁻⁶) and is at least on par with a strong
   gradient-boosting baseline.
2. **Spatial transfer.** In 4-fold leave-group-out — every station held out once —
   the model transfers to unseen basins (skill +0.13 / +0.14 / +0.23 over
   persistence at 1 / 3 / 7 days), a claim a single-site study cannot make.
3. **Calibrated uncertainty.** Conformalised intervals are near-nominal across the
   population (PICP ≈ 0.90; 89–97 % of stations within ±0.05 of target).
4. **Component attribution.** Seeded ablations with paired tests show which model
   parts earn their place.

We deliberately also report **three negative results**: no point-accuracy gain on a
near-deterministic reservoir cascade, a flow-dependent thermal memory that does not
generalise beyond it, and no robust cost–loss decision-value advantage over a strong
deterministic persistence warning. We believe this honesty — and the demonstration
that a method's advantage must be established on a large, diverse sample rather than
a single site — is itself a useful corrective for the field, and a natural fit for
the methodological standards of *Journal of Hydrology*.

All code, the fixed evaluation protocol, the assembled datasets, unit tests, and a
one-command reproduction are provided; every headline claim was checked by an
adversarial internal review against the result tables, with a finding-by-finding
disposition included.

We confirm that this manuscript is original, has not been published previously, and
is not under consideration elsewhere. All authors have approved the submission and
declare no competing interests. The data are public-domain/open-licensed and the
code is available for review.

Thank you for considering our work.

Sincerely,
The authors
