# SI19 — development-period spatial analysis and robustness probes

Relocated from manuscript Sections 4.4 and 4.5 by the advisory length plan (Batch A): development-window diagnostics that the independent-window Sections 4.4–4.6 supersede.

### 4.4 Whole-region holdout reduces the reported transfer skill in the development window

How much of a reported spatial transfer survives when the map is partitioned
honestly? On the development window, between a quarter and a third of it does
not; the independent-window re-test of Section 4.7 measures the same contrast
with a factorial design.

| Arm | Design | 1 d | 3 d | 7 d |
|---|---|---:|---:|---:|
| Temporal development | seen stations, 2019–2020 | +0.203 | +0.187 | +0.251 |
| Random held-site warm start | 4 folds; held-site history in preprocessing | +0.178 | +0.172 | +0.241 |
| Held-region gauged transfer | leave-HUC2-region-out | +0.155 | +0.116 | +0.155 |

*Development-period median station skill versus persistence, dimensionless.* Against damped
persistence the same three arms give +0.168 / +0.076 / +0.038 (temporal),
+0.145 / +0.061 / +0.030 (random held-site), and +0.147 / +0.086 / +0.075
(held-region). The three-day skill against persistence falls from +0.187 to
+0.172 when sites are held out at random and to +0.116 when whole regions are,
and the seven-day figure falls from +0.251 to +0.241 to +0.155. The gap between
the second and third rows is a development-window measure of how much a random
spatial split flatters a model on this panel: the random split retains 92% of the
temporal-arm three-day skill, the regional split 62%; Section 4.7 re-tests this
finding on the independent 2021–2023 window with a matched 2×2 factorial design
that separates geometry from local adaptation and repeats the random split over
five seeds.

The mechanism is neighbourhood, and the geometry supports that reading. Under
whole-region holdout the mean distance from a held-out station to its nearest
training gauge is 289 km, whereas in the intact cohort 19 of 120 stations have
another retained station within 10 km and 38 share a hydrologic unit code with
another retained station. A random held-site split on such a panel measures
something much closer to interpolation between instrumented neighbours than to
transfer to an uninstrumented region.

The ranking of models is unchanged by the harder partition, and the margins
widen. In the held-region arm the global tree ensemble again attains lower
station-median RMSE than ThermoRoute (0.652 against 0.676 °C at 1 day, 1.391
against 1.428 at 3 days, 1.786 against 1.860 at 7 days), with paired median
differences of +0.031 [+0.023, +0.040], +0.038 [+0.029, +0.050], and +0.067
[+0.049, +0.080] °C and ThermoRoute win rates near 0.16. The global LSTM, with
its station embedding disabled in this arm, gives 0.679, 1.445, and 1.876 °C.
Since the spatial partition moves the answer this much, the next question is
whether the remaining skill is spatially and seasonally uniform.





---

### 4.5 Skill is regionally uniform, and interval coverage is bought with width

It is uniform, and this is the one place where the constrained model's behavior
is unremarkable in a useful way. Region-weighted skill against persistence — the
mean of the 15 per-HUC2 medians — is +0.204, +0.189, and +0.253 at 1, 3, and 7
days, essentially unchanged from the pooled medians of +0.203, +0.187, and
+0.251, so no single region carries the headline. Per-region medians against
persistence at 1 day range from +0.131 (HUC2:10) to +0.285 (HUC2:18) and at 7
days from +0.217 (HUC2:06) to +0.290 (HUC2:09). Stratifying by drainage area into
three groups of roughly 39 stations gives 1-day medians of +0.190, +0.214, and
+0.231 from small to large. Against damped persistence the seven-day per-region
medians compress to a range of +0.012 to +0.077 — the same message as Section
4.1, restated spatially: what varies across regions is small once the seasonal
reference is doing its work.

Interval behavior tells the complementary story about what calibration costs.
Split-conformal intervals on the 249,072 development keys attain an empirical
marginal coverage of 0.909 against a nominal 0.90, with a mean width of 3.87 °C
and a mean interval score of 4.93; per lead, coverage is 0.905, 0.910, and 0.912
at widths of 2.01, 4.22, and 5.38 °C, and 0.918 at a width of 3.50 °C on the
warm-season q90-exceedance tail. Block-calibrating the maximum nonconformity
score over seven-row blocks raises coverage to 0.981 but widens the interval to
5.74 °C — coverage bought with width, not sharpness; delayed adaptive-conformal
variants track nominal coverage (0.892–0.903) but produce unbounded widths in
several slices (details in SI09). None of these figures is a conditional-
coverage statement, and width and interval score are reported beside every
coverage figure so that adaptive calibration is not presented as free.

Input perturbations locate the model's dependence in the same two channels the
ablations pointed to. Gaussian sensor noise at 0.25 and 0.5 training standard
deviations degrades 1-day RMSE from 0.631 to 1.621 and 2.999 °C (+147% and
+360%); missing-forcing blocks of 3, 7, and 14 days cost about +0.13 °C at 1 day
and +0.04 °C at 7 days; air-temperature offsets of ±2 training standard
deviations cost +0.11 to +0.15 °C at 1 day; and multiplying discharge by 0.5 or 2
changes 1-day RMSE by at most +0.003 °C. These are synthetic data-corruption
probes, not climate projections, physically coherent scenarios, or
deployment-safety tests. Scale insensitivity is not absence of information:
retraining the gradient-boosted tree without the discharge channel at all
degrades station-median RMSE by +0.042 °C at one day (0.622 versus 0.589 °C),
+0.034 °C at three days, and +0.009 °C at seven days, with the no-flow model
winning at only 3% of stations at one day and 37% at seven days
(`outputs/final/flow_ablation_effects.parquet`). Discharge therefore carries
small but systematic information at the shortest lead, so its exclusion from
the cohort (Section 2.1) trades a real, modest channel against much wider
spatial coverage; quantifying that trade-off by re-deriving a no-flow core
cohort from the candidate registry is recorded as future work (protocol P2).
