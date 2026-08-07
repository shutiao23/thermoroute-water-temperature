# SI03 — model identities, units and identifiability limits

**Status:** scaffold finalized; empirical values `[pending computation]`.

This SI records only identities and fixed architecture descriptions already
stated in the PRE manuscript or Figure-1 skeleton. It contains no fitted
parameter, learned weight, performance quantity or completion receipt. Any such
value is `[pending computation]`.

## Point identity and bound

The canonical Figure-1 identity is

\[
\hat{y}_{t+h}
=
A_{t+h}
+
\delta\,\tanh\!\Bigl(
\frac{P_{t+h}-A_{t+h}+r_{\theta,t+h}}{\delta}
\Bigr),
\qquad
\delta=\texttt{delta\_scale}.
\]

Here, \(A_{t+h}\) is the frozen damped-persistence anchor, \(P_{t+h}\) is the
learned thermal proposal, \(r_{\theta,t+h}\) is the neural residual and
\(\hat{y}_{t+h}\) is the point forecast. The PRE configuration sets
`delta_scale = 1.0 °C`. With finite \(\delta\), the point output lies in
\([A_{t+h}-\delta, A_{t+h}+\delta]\). These are algebraic design facts from
`paper/FIGURE1_AND_SI_SKELETON.md` §1b and
`paper/ThermoRoute_paper.md` §§3.3, 4.1.

**Non-safety interpretation:** the \(\pm\delta\) envelope bounds deviation from
the anchor only. It does not bound absolute error, event-tail risk, interval
width, post-shift behavior, deployment safety, or regulatory safety.

## Fixed architecture descriptions

| Component | PRE description | Source and role |
|---|---|---|
| anchor | damped persistence moves last observed \(y_t\) toward frozen seasonal climatology \(c_{t+h}\) at a fixed rate | PRE manuscript §3.1; descriptive identity only |
| learned proposal | flow- and season-conditioned relaxation proposal | PRE manuscript §3.1; not a residence-time or heat-transfer measurement |
| router | seven variables at lags 0–14 | PRE manuscript §3.2; input-allocation geometry |
| sequence buffer | 32-day tensor | PRE manuscript §3.2; construction buffer, not effective memory |
| TCN | two blocks, kernel three, strictly left-looking; theoretical seven-step receptive field | PRE manuscript §3.2; architecture geometry |
| mixture | mixture-of-experts representation combination | PRE manuscript §3.2; no river-network interpretation |

The PRE manuscript does not provide a symbolic learned-relaxation, router-weight,
TCN-kernel, mixture-gate, or Platt-coefficient equation suitable for independent
reconstruction here. This scaffold must not invent one. A later rendered equation
requires a frozen implementation/configuration binding and a source hash.

## Quantile, CQR and event-probability contract

The learned models emit separate point and pinball-trained \(q_{0.05}\),
\(q_{0.50}\), and \(q_{0.95}\) heads. After equal-weight member averaging, CQR
is fitted on 2018 only. Its deployed offset rule is:

\[
q^+ = \max(\mathrm{raw\_qhat},0),\qquad
[q_{0.05}-q^+,\ q_{0.95}+q^+],
\]

with \(q_{0.50}\) unchanged. Thus CQR may retain or widen, but cannot shrink,
the nominal interval. Temporal offsets are station-by-horizon; external offsets
are pooled by horizon. One Platt calibrator per horizon is fitted on 2018 only;
the temporal event threshold is the station-specific 2006–2015 q90, while the
frozen seasonal event-reference fit spans 2006–2018. These are frozen procedure
facts from `paper/ThermoRoute_paper.md` §3.3, not fitted values.

Reported coverage is empirical marginal coverage only; three-quantile pinball is
not CRPS, and no SI may call it conditional coverage. All probability metrics,
interval widths, calibration parameters and reliability coordinates remain
`[pending computation]` until receipt-bound.
