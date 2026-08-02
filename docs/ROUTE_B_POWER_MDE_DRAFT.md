# Route B power, MDE and PSU assurance draft

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **PRELABEL DESIGN ONLY / NO TARGET OUTCOME INPUT** |
| Dependency | Final task arm, estimand, hypothesis family, scientific margin and year scope require user choice |

## 1. Estimand and screening formula

On exact common keys, compute model RMSE and the paired difference within site,
network PSU, year and horizon. Negative favors the candidate. First estimate the
eligible-site mean inside PSU×year using the frozen within-PSU design, then weight
years equally and PSUs by normalized inverse inclusion probability:

\[
\widehat\theta_h=
\sum_{k=1}^{K}w_kY_k^{-1}\sum_y D_{kyh},\qquad
w_k\propto 1/\pi_k,\quad \sum_kw_k=1.
\]

For a self-weighting one-site-per-PSU design, `w_k=1/K`. Otherwise the simulation
and replicate bootstrap use the exact two-stage component/site weights; an
unweighted sample mean is descriptive only.

Choose one of two claims before simulation:

- fixed-three-year estimand: inference is restricted to the named untouched
  years, with PSU and within-year synchronized time-block uncertainty;
- year-superpopulation estimand: crossed PSU×year/block uncertainty. Three years
  provide weak year-level evidence, so this claim should add years or remain
  tightly qualified.

A planning approximation is:

\[
SE^2 \approx \sigma_N^2/K + \sigma_Y^2/Y + \sigma_{NY}^2/(KY),
\]

\[
MDE \approx (t_{1-\alpha^*,K-1}+z_{1-\beta})SE.
\]

The detectable distance from the testing boundary is
`δ = margin − alternative effect`. A superiority boundary may be zero. A
non-inferiority margin is a scientific/decision quantity and cannot be derived
from this power formula, sensor storage resolution or Route-A's +0.05 °C ceiling.

As an analytic screening check for 80% power, standardized MDE is approximately
0.45/0.37/0.35/0.34/0.32 for K=30/45/50/55/60 at one-sided α=0.05, and
0.58/0.47/0.45/0.43/0.41 at α=0.01. These values are not the final decision rule;
the registered simulation must reproduce the planned RMSE estimator and UQ.

## 2. Prelabel simulation grid

Use a curated factorial/sensitivity design rather than silently discarding hard
full-grid cells:

| Parameter | Grid |
| --- | --- |
| PSU K | 30, 35, 40, 45, 50, 55, 60 |
| target years Y | 1, 2, 3, 5 |
| required power | 0.80, 0.90 |
| α | 0.05 and family-adjusted conservative value |
| standardized effect δ/σ | null and 0.1–0.8; also report °C |
| network ICC | 0, .05, .10, .20, .35, .50 |
| year ICC | 0, .01, .025, .05, .10 |
| PSU attrition | 0%, 10%, 20%, 30%, 40% |
| attrition correlation | beta-binomial ICC 0, .05, .15, .30 |
| informative attrition shift | 0, ±0.25σ, ±0.5σ |
| sites per PSU | 1, 2, 4, 8 |
| PSU size CV | 0, .5, 1, 2 |
| size–effect correlation | −.5, 0, .5 |
| daily residual AR(1) | 0, .3, .6, .8 |
| candidate/reference error correlation | .5, .7, .9 |
| error distribution | Normal, t5, skew/heavy-tail mixture |
| synchronized calendar block | 7, 14, 28, 56 days |

Every simulated UQ draw reconstructs daily candidate/reference errors and
recomputes RMSE. Resampling pre-aggregated RMSE cannot stand in for temporal UQ.

Input parameter evidence may come from external literature, synthetic stress
bounds and clearly labelled pre-Route-B development data. Route-A opening effects,
variance, attrition or favorable subgroups may never update this grid or K rule.

## 3. PSU dependence and selection

Build a frozen dependence graph using upstream catchment overlap, either-direction
flow connectivity, shared reservoir/backwater/management domain and predeclared
spatial/climate buffers. Each connected component receives one PSU identity. It
must not be split merely to obtain K≥30.

Exclude Route-A sites and any Route-A-connected/overlapping PSU. Multiple sites
inside one PSU may improve within-PSU precision but remains one graph component;
analysis follows the frozen two-stage design weight.

Select:

\[
K_0=\min_{45\le K\le60}
\{K:\Pr(K_{reportable}\ge30)\ge A,\ Power\ge P,\ CI\ halfwidth\le W\}.
\]

The user freezes assurance `A`, power `P` and precision `W`. Under deterministic
loss only, K=45/50/55/60 can lose at most 15/20/25/30 PSUs and retain 30; correlated
attrition must instead use the beta-binomial assurance calculation.

Every formal comparison/horizon must also pass:

- `K_reportable >= 30`;
- Kish `K_effective / K_reportable >= 0.75`;
- largest weight `< 0.25`.

If K=60 does not meet the frozen assurance/power/precision rule, do not relax the
cluster gate or change the margin. Add untouched years, expand the legal sampling
frame or stop the strong-inference claim.

## 4. Artifacts

```text
sampling_frame.parquet         # site/reach/catchment/PSU/exclusion/stratum/random rank
psu_registry.json              # graph version/hash, edge rules, Route-A leakage, balance
power_design.json              # estimand/year scope/hypotheses/margin source/alpha/grid/K rule
power_grid.parquet             # scenario, parameters, analytic MDE, simulated power/coverage/MCSE
power_receipt.json             # self-hash and complete input/output bindings
route_b_preregistration.json + independent seal/timestamp
```

## 5. Acceptance

- target-label access attestation is false and target paths are absent;
- PSU construction and random selection replay exactly;
- normalized two-stage design weights and year-equal aggregation reproduce the
  estimator (equal PSU weights only for a verified self-weighting design);
- registered simulation attains predeclared Monte Carlo SE for Type-I error,
  interval coverage, assurance and power;
- every hard/null/informative-attrition scenario remains in the output;
- no result-dependent margin, K, block, PSU edge or hypothesis change is allowed.
