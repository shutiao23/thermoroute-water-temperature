"""Prequential conformal diagnostics with delayed target feedback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import conformal_quantile


def delayed_aci(calibration_scores: np.ndarray, evaluation: pd.DataFrame, *,
                alpha: float = 0.10, gamma: float = 0.02,
                issue_col: str = "issue_date", target_date_col: str = "target_date",
                target_col: str = "y_true", lower_col: str = "q05",
                upper_col: str = "q95") -> pd.DataFrame:
    """Run ACI without using a label before its target date has arrived.

    All forecasts sharing one issue date use the same pre-update state.  Feedback
    whose target date equals the current issue date is processed before issuing the
    new forecast, matching an end-of-day observation/next-issue convention.
    """
    required = {issue_col, target_date_col, target_col, lower_col, upper_col}
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"ACI evaluation frame missing: {sorted(missing)}")
    if not 0 < alpha < 1 or gamma <= 0:
        raise ValueError("alpha must be in (0,1) and gamma must be positive")
    frame = evaluation.copy()
    frame[issue_col] = pd.to_datetime(frame[issue_col])
    frame[target_date_col] = pd.to_datetime(frame[target_date_col])
    if (frame[target_date_col] <= frame[issue_col]).any():
        raise ValueError("every ACI target must be strictly after its issue date")
    frame = frame.sort_values([issue_col, target_date_col]).copy()
    frame["aci_alpha"] = np.nan
    frame["aci_offset"] = np.nan
    frame["aci_lower"] = np.nan
    frame["aci_upper"] = np.nan
    frame["aci_covered"] = False
    frame["aci_width"] = np.nan
    frame["aci_interval_score"] = np.nan
    frame["feedback_count"] = 0

    state = float(alpha)
    feedback_seen = 0
    pending: list[tuple[pd.Timestamp, bool]] = []
    for issue_date, indices in frame.groupby(issue_col, sort=True).groups.items():
        issue_date = pd.Timestamp(issue_date)
        arrived = [entry for entry in pending if entry[0] <= issue_date]
        pending = [entry for entry in pending if entry[0] > issue_date]
        for _, covered in sorted(arrived, key=lambda entry: entry[0]):
            error = 0.0 if covered else 1.0
            state = float(np.clip(state + gamma * (alpha - error), 1e-3, 0.5))
            feedback_seen += 1

        offset = conformal_quantile(calibration_scores, state)
        for index in indices:
            row = frame.loc[index]
            lower = float(row[lower_col] - offset)
            upper = float(row[upper_col] + offset)
            target = float(row[target_col])
            covered = lower <= target <= upper
            width = upper - lower
            penalty = 0.0
            if target < lower:
                penalty = 2.0 / alpha * (lower - target)
            elif target > upper:
                penalty = 2.0 / alpha * (target - upper)
            frame.loc[index, [
                "aci_alpha", "aci_offset", "aci_lower", "aci_upper",
                "aci_covered", "aci_width", "aci_interval_score", "feedback_count",
            ]] = [state, offset, lower, upper, covered, width, width + penalty, feedback_seen]
            pending.append((pd.Timestamp(row[target_date_col]), bool(covered)))
    return frame.sort_index()
