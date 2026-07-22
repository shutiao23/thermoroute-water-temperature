"""Significance tools that respect temporal autocorrelation.

Daily water temperature is strongly autocorrelated, so i.i.d. bootstrap and
naive paired t-tests understate uncertainty.  We use a moving-block bootstrap
for confidence intervals and a Diebold-Mariano test with an autocorrelation
(HAC) variance for pairwise model comparison.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


EXPLORATORY_CLUSTER_SENSITIVITY_ROLE = (
    "EXPLORATORY_SENSITIVITY_NOT_A_CONFIRMATORY_TEST_OR_DECISION"
)
DEFAULT_SMALL_CLUSTER_THRESHOLD = 30
DEFAULT_DOMINANT_CLUSTER_SHARE = 0.25
DEFAULT_LOW_EFFECTIVE_CLUSTER_FRACTION = 0.75


def _validated_cluster_inputs(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    null_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite effects and non-missing cluster identifiers."""
    values = np.asarray(effects, dtype=float)
    groups = np.asarray(clusters)
    if values.ndim != 1 or groups.ndim != 1:
        raise ValueError("effects and clusters must be one-dimensional")
    if len(values) != len(groups):
        raise ValueError("effects and clusters must have identical lengths")
    if not np.isfinite(null_margin):
        raise ValueError("null_margin must be finite")
    valid = np.isfinite(values) & pd_notna(groups)
    values, groups = values[valid], groups[valid]
    if len(values) == 0:
        raise ValueError("no finite paired effects")
    return values, groups


def _reduce_effect(values: np.ndarray, statistic: str) -> float:
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    raise ValueError("statistic must be 'median' or 'mean'")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def cluster_bootstrap_paired_effect(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    statistic: str = "median",
    n_boot: int = 10000,
    seed: int = 0,
    null_margin: float = 0.0,
) -> dict[str, float]:
    """Resample whole hydrologic clusters for an effect and percentile CI.

    Effects use the convention ``candidate - reference``; negative is better.
    This function intentionally does not manufacture a hypothesis-test p-value
    from bootstrap tail counts.  Route A uses :func:`cluster_sign_flip_pvalue`
    for its separate, assumption-explicit randomisation test.  ``null_margin``
    is retained only for API compatibility with preregistered callers.
    """
    effects, clusters = _validated_cluster_inputs(
        effects, clusters, null_margin=null_margin
    )
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    _reduce_effect(effects, statistic)

    groups = {cluster: effects[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot)
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        bootstrap[index] = _reduce_effect(
            np.concatenate([groups[cluster] for cluster in sampled]), statistic
        )
    point = _reduce_effect(effects, statistic)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "effect": point,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_stations": int(len(effects)),
        "n_clusters": int(len(unique)),
    }


def pd_notna(values: np.ndarray) -> np.ndarray:
    """Small dependency-free missing-value mask for numeric/string cluster IDs."""
    return np.asarray([value is not None and str(value).lower() != "nan" for value in values])


def equivalence_decision(ci_low: float, ci_high: float, margin: float) -> bool:
    """Two-sided equivalence requires the complete interval inside ±margin."""
    if margin <= 0:
        raise ValueError("equivalence margin must be positive")
    return bool(ci_low > -margin and ci_high < margin)


def noninferiority_decision(ci_high: float, margin: float) -> bool:
    """Candidate is non-inferior when the upper loss-difference bound is below margin."""
    if margin <= 0:
        raise ValueError("non-inferiority margin must be positive")
    return bool(ci_high < margin)


def cluster_sign_flip_pvalue(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    null_margin: float = 0.0,
    statistic: str = "median",
    n_randomisations: int = 50000,
    seed: int = 0,
    exact_max_clusters: int = 20,
) -> float:
    """One-sided cluster sign-flip test for a lower paired effect.

    The paired differences use ``candidate - reference`` and the alternative is
    that their location is below ``null_margin``.  One sign is assigned to every
    complete HUC/spatial cluster, preserving within-cluster dependence.  Up to
    ``min(exact_max_clusters, 20)`` all :math:`2^K` sign vectors are enumerated;
    larger exploratory registries use the seeded Monte-Carlo fallback.  The test
    assumes independent whole-cluster sign reversals and joint sign symmetry of
    each cluster vector under the null; the assumption is explicit and is
    preferable to treating daily or station observations as independent.
    """
    values, groups = _validated_cluster_inputs(
        effects, clusters, null_margin=null_margin
    )
    values = values - float(null_margin)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("cluster sign-flip needs at least two clusters")
    if int(exact_max_clusters) < 2:
        raise ValueError("exact_max_clusters must be at least two")
    exact_limit = min(int(exact_max_clusters), 20)
    observed = _reduce_effect(values, statistic)
    cluster_index = np.searchsorted(unique, groups)
    # Route A has only 15 frozen HUC2 clusters.  Enumerating every sign vector
    # avoids Monte-Carlo error and makes the p-value independent of a PRNG.
    # Retain a streamed Monte-Carlo fallback for larger exploratory registries.
    if len(unique) <= exact_limit:
        count = 0
        configurations = 1 << len(unique)
        for start in range(0, configurations, 4096):
            stop = min(start + 4096, configurations)
            codes = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = (codes >> np.arange(len(unique), dtype=np.uint64)) & 1
            signs = bits.astype(float) * 2.0 - 1.0
            signed = signs[:, cluster_index] * values[None, :]
            simulated = np.median(signed, axis=1) if statistic == "median" else np.mean(
                signed, axis=1
            )
            count += int(np.sum(simulated <= observed + 1e-15))
        return float(count / configurations)

    rng = np.random.default_rng(seed)
    count = 0
    # Use the conservative add-one Monte-Carlo correction and stream batches so
    # the fallback has bounded memory for many clusters.
    remaining = int(n_randomisations)
    if remaining < 100:
        raise ValueError("n_randomisations must be at least 100")
    while remaining:
        batch_size = min(4096, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch_size, len(unique)))
        signed = signs[:, cluster_index] * values[None, :]
        simulated = np.median(signed, axis=1) if statistic == "median" else np.mean(
            signed, axis=1
        )
        count += int(np.sum(simulated <= observed + 1e-15))
        remaining -= batch_size
    return float((count + 1) / (n_randomisations + 1))


def cluster_inference_sensitivity(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    null_margin: float = 0.0,
    statistic: str = "median",
    n_randomisations: int = 50000,
    seed: int = 0,
    exact_max_clusters: int = 20,
    small_cluster_threshold: int = DEFAULT_SMALL_CLUSTER_THRESHOLD,
    dominant_cluster_share: float = DEFAULT_DOMINANT_CLUSTER_SHARE,
    low_effective_cluster_fraction: float = DEFAULT_LOW_EFFECTIVE_CLUSTER_FRACTION,
) -> dict[str, object]:
    """Describe cluster leverage without changing confirmatory inference.

    Effects must be one station-level paired estimate per row and use the
    ``candidate - reference`` convention.  Thus a value below ``null_margin``
    favours the candidate under the lower-is-better alternative.  Every point
    and leave-one-cluster-out (LOCO) statistic remains station weighted; HUC2 is
    used only as the dependence/randomisation unit.

    The sign-flip diagnostic assumes independent whole-cluster sign reversals
    and *joint sign symmetry of each complete cluster effect vector* after
    subtracting ``null_margin``.  It is enumerated exactly for at most 20
    clusters and otherwise uses a reproducible seeded Monte-Carlo approximation.
    Its p-value is deliberately labelled a sensitivity result: it is not an
    extra confirmatory test, is not added to Holm's family, and must not replace
    the frozen five-test rule.  The inverse-Herfindahl effective cluster count is
    a cluster-size-concentration diagnostic, not a replacement degrees-of-
    freedom formula for the nonlinear station median.
    """
    values, groups = _validated_cluster_inputs(
        effects, clusters, null_margin=null_margin
    )
    point = _reduce_effect(values, statistic)
    unique, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    n_clusters = int(len(unique))
    n_stations = int(len(values))
    shares = counts.astype(float) / n_stations
    largest_share = float(np.max(shares))
    effective_clusters = float(1.0 / np.sum(shares ** 2))
    effective_fraction = float(effective_clusters / n_clusters)
    cluster_size_mean = float(np.mean(counts))
    cluster_size_cv = float(np.std(counts, ddof=0) / cluster_size_mean)

    if small_cluster_threshold < 2:
        raise ValueError("small_cluster_threshold must be at least two")
    if not 0.0 < dominant_cluster_share <= 1.0:
        raise ValueError("dominant_cluster_share must be in (0, 1]")
    if not 0.0 < low_effective_cluster_fraction <= 1.0:
        raise ValueError("low_effective_cluster_fraction must be in (0, 1]")

    leave_one_cluster_out: list[dict[str, object]] = []
    loco_effects: list[float] = []
    for cluster_index, cluster in enumerate(unique):
        keep = inverse != cluster_index
        held_out_count = int(np.sum(~keep))
        if not np.any(keep):
            loco_effect = None
            centered = None
        else:
            loco_effect = _reduce_effect(values[keep], statistic)
            centered = float(loco_effect - null_margin)
            loco_effects.append(loco_effect)
        leave_one_cluster_out.append({
            "held_out_cluster": str(cluster),
            "held_out_station_count": held_out_count,
            "remaining_station_count": int(np.sum(keep)),
            "remaining_cluster_count": n_clusters - 1,
            "station_weighted_effect": loco_effect,
            "effect_minus_null_margin": centered,
        })

    point_centered = float(point - null_margin)
    if not loco_effects:
        loco_direction = "NOT_ESTIMABLE_SINGLE_CLUSTER"
        loco_direction_stable = False
        loco_min = loco_max = loco_centered_min = loco_centered_max = None
        max_abs_loco_shift = None
    else:
        centered_loco = np.asarray(loco_effects, dtype=float) - float(null_margin)
        loco_min = float(np.min(loco_effects))
        loco_max = float(np.max(loco_effects))
        loco_centered_min = float(np.min(centered_loco))
        loco_centered_max = float(np.max(centered_loco))
        max_abs_loco_shift = float(np.max(np.abs(np.asarray(loco_effects) - point)))
        if np.all(centered_loco < 0.0):
            loco_direction = "ALL_LOCO_EFFECTS_BELOW_NULL_MARGIN"
        elif np.all(centered_loco > 0.0):
            loco_direction = "ALL_LOCO_EFFECTS_ABOVE_NULL_MARGIN"
        else:
            loco_direction = "LOCO_EFFECTS_CROSS_OR_TOUCH_NULL_MARGIN"
        loco_direction_stable = bool(
            (point_centered < 0.0 and np.all(centered_loco < 0.0))
            or (point_centered > 0.0 and np.all(centered_loco > 0.0))
        )

    warnings: list[str] = []
    if n_clusters < 2:
        warnings.append("SINGLE_CLUSTER_RANDOMIZATION_NOT_ESTIMABLE")
    if n_clusters < small_cluster_threshold:
        warnings.append(f"SMALL_CLUSTER_COUNT_LT_{small_cluster_threshold}")
    if largest_share >= dominant_cluster_share:
        warnings.append("DOMINANT_CLUSTER_SHARE_AT_OR_ABOVE_THRESHOLD")
    if effective_fraction < low_effective_cluster_fraction:
        warnings.append("LOW_EFFECTIVE_CLUSTER_FRACTION")
    if loco_direction == "LOCO_EFFECTS_CROSS_OR_TOUCH_NULL_MARGIN":
        warnings.append("LOCO_DIRECTION_CROSSES_OR_TOUCHES_NULL_MARGIN")

    if n_clusters < 2:
        randomisation_method = "NOT_ESTIMABLE_SINGLE_CLUSTER"
        sign_flip_p: float | None = None
        draws_evaluated = 0
        exact_enumeration = False
        monte_carlo_standard_error: float | None = None
    else:
        if int(exact_max_clusters) < 2:
            raise ValueError("exact_max_clusters must be at least two")
        exact_enumeration = n_clusters <= min(int(exact_max_clusters), 20)
        randomisation_method = (
            "EXACT_WHOLE_CLUSTER_SIGN_FLIP_ENUMERATION"
            if exact_enumeration
            else "SEEDED_MONTE_CARLO_WHOLE_CLUSTER_SIGN_FLIP"
        )
        draws_evaluated = (
            1 << n_clusters if exact_enumeration else int(n_randomisations)
        )
        sign_flip_p = cluster_sign_flip_pvalue(
            values,
            groups,
            null_margin=null_margin,
            statistic=statistic,
            n_randomisations=n_randomisations,
            seed=seed,
            exact_max_clusters=exact_max_clusters,
        )
        monte_carlo_standard_error = (
            0.0
            if exact_enumeration
            else float(np.sqrt(
                sign_flip_p * (1.0 - sign_flip_p) / (n_randomisations + 1)
            ))
        )

    no_strong_inference = bool(warnings)
    return {
        "role": EXPLORATORY_CLUSTER_SENSITIVITY_ROLE,
        "formal_five_test_decision_rule_modified": False,
        "effect_convention": "station_RMSE_candidate_minus_reference_lower_is_better",
        "estimand_weighting": "equal_station",
        "dependence_unit": "whole_cluster",
        "statistic": statistic,
        "null_margin": float(null_margin),
        "alternative": "station_weighted_effect_below_null_margin",
        "sign_flip_null_assumption": (
            "independent whole-cluster sign reversals and joint sign symmetry of "
            "each complete cluster effect vector after subtracting null_margin"
        ),
        "point_effect": point,
        "point_effect_minus_null_margin": point_centered,
        "n_stations": n_stations,
        "n_clusters": n_clusters,
        "cluster_size_min": int(np.min(counts)),
        "cluster_size_median": float(np.median(counts)),
        "cluster_size_max": int(np.max(counts)),
        "cluster_size_cv": cluster_size_cv,
        "largest_cluster_share": largest_share,
        "effective_cluster_count_inverse_herfindahl": effective_clusters,
        "effective_cluster_fraction": effective_fraction,
        "randomisation_method": randomisation_method,
        "randomisation_seed": None if exact_enumeration else int(seed),
        "randomisation_draws_evaluated": draws_evaluated,
        "exact_enumeration": exact_enumeration,
        "sign_flip_p_one_sided_sensitivity": sign_flip_p,
        "monte_carlo_standard_error": monte_carlo_standard_error,
        "loco_effect_min": loco_min,
        "loco_effect_max": loco_max,
        "loco_effect_minus_null_margin_min": loco_centered_min,
        "loco_effect_minus_null_margin_max": loco_centered_max,
        "loco_max_abs_shift_from_full": max_abs_loco_shift,
        "loco_direction": loco_direction,
        "loco_direction_stable": loco_direction_stable,
        "leave_one_cluster_out": leave_one_cluster_out,
        "warning_codes": warnings,
        "inference_strength": (
            "NO_STRONG_INFERENCE" if no_strong_inference else "NO_OBVIOUS_CLUSTER_LEVERAGE"
        ),
    }


def moving_block_bootstrap_ci(
    errors_sq: np.ndarray, block: int = 30, n_boot: int = 2000,
    seed: int = 0, stat: str = "rmse",
) -> tuple[float, float, float]:
    """Bootstrap CI for RMSE/MAE from a series of per-day squared/abs errors.

    ``errors_sq`` should be squared errors for RMSE or absolute errors for MAE.
    Returns ``(point, lo95, hi95)``.
    """
    rng = np.random.default_rng(seed)
    n = len(errors_sq)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    point = float(np.sqrt(errors_sq.mean())) if stat == "rmse" else float(errors_sq.mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, starts_max + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        samp = errors_sq[idx]
        boots[b] = np.sqrt(samp.mean()) if stat == "rmse" else samp.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def block_bootstrap_station_avg_rmse(
    err2_by_station: dict[str, np.ndarray], block: int = 30, n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the *station-averaged* RMSE (the headline aggregation).

    Each station's per-day squared errors are block-resampled independently;
    per-station RMSE is computed, then averaged over stations. This reproduces
    the reported point estimate (mean over stations of per-station RMSE) and,
    crucially, does NOT average predictions across stations (which would cancel
    independent errors and deflate the RMSE).
    """
    rng = np.random.default_rng(seed)
    stations = list(err2_by_station)
    point = float(np.mean([np.sqrt(err2_by_station[s].mean()) for s in stations]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        per_station = []
        for s in stations:
            e = err2_by_station[s]
            n = len(e)
            n_blocks = int(np.ceil(n / block))
            starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
            idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
            per_station.append(np.sqrt(e[idx].mean()))
        boots[b] = np.mean(per_station)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def diebold_mariano(
    err_a: np.ndarray, err_b: np.ndarray, h: int = 1, loss: str = "sq",
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy (model A vs B).

    Negative DM statistic ⇒ model A has the smaller loss.  Uses a Newey-West HAC
    variance with bandwidth ``h-1`` (small-sample Harvey correction applied).
    Returns ``(dm_stat, two_sided_p)``.
    """
    if loss == "sq":
        d = err_a ** 2 - err_b ** 2
    else:
        d = np.abs(err_a) - np.abs(err_b)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * cov
    var = var / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    # Harvey, Leybourne & Newbold small-sample correction
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= adj
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)
