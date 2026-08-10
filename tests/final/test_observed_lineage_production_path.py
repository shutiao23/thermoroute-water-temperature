"""Production-data regression for the v5 observed-lineage preprocessing path.

This module intentionally stops at the exact formal evaluation binding.  It
does not fit a model, read predictions, or make assertions about model scores.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.final.run_forcing_ladder_v5_observed as V5  # noqa: E402
from thermoroute import config as C  # noqa: E402


@dataclass(frozen=True)
class _ProductionPath:
    raw: V5.RawLabelPanel
    stations: tuple[str, ...]
    reference: pd.DataFrame
    preprocessing: V5.ObservedPreprocessing
    imputed: V5.ImputedFeaturePanel


@pytest.fixture(scope="module")
def production_path() -> Iterator[_ProductionPath]:
    """Run the real pinned inputs through the pre-model production path once."""

    original_stations = tuple(C.STATIONS)

    def forbidden_training(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("observed-lineage regression crossed into model training")

    # Keep the guard active for fixture construction and every test that uses
    # it.  Preprocessing fits are in scope; model fitting and scoring are not.
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(V5, "_lgb_fit", forbidden_training)
        guard.setattr(V5, "fit_tree_cell", forbidden_training)
        try:
            raw, stations, _station_registry = V5.load_pinned_raw_panel()
            reference = V5.load_pinned_reference_registry()
            preprocessing = V5.fit_observed_preprocessing(raw, stations)
            imputed = V5.impute_feature_panel(raw, preprocessing.imputer)
            yield _ProductionPath(
                raw=raw,
                stations=stations,
                reference=reference,
                preprocessing=preprocessing,
                imputed=imputed,
            )
        finally:
            C.STATIONS = original_stations


def _identity_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["site_id", "issue_date", "target_date"]])


def _raw_issue_target_grid(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    issues = raw[["site_id", "DATE", "WTEMP_observed"]].rename(
        columns={"DATE": "issue_date", "WTEMP_observed": "issue_raw_observed"}
    )
    issues = issues.copy()
    issues["target_date"] = issues["issue_date"] + pd.Timedelta(days=horizon)
    targets = raw[["site_id", "DATE", "WTEMP", "WTEMP_observed"]].rename(
        columns={
            "DATE": "target_date",
            "WTEMP": "target_raw_wtemp",
            "WTEMP_observed": "target_raw_observed",
        }
    )
    return issues.merge(
        targets,
        on=["site_id", "target_date"],
        how="left",
        sort=False,
        validate="many_to_one",
    )


def test_real_raw_finiteness_masks_survive_production_imputation(
    production_path: _ProductionPath,
) -> None:
    raw = production_path.raw.frame
    imputed = production_path.imputed.frame

    assert production_path.imputed.raw_source_sha256 == production_path.raw.source_sha256
    assert production_path.imputed.raw_identity_sha256 == production_path.raw.identity_sha256
    assert production_path.imputed.observed_mask_sha256 == (
        production_path.raw.observed_mask_sha256
    )
    pd.testing.assert_frame_equal(
        imputed[["site_id", "DATE"]],
        raw[["site_id", "DATE"]],
        check_exact=True,
    )

    missing_counts: dict[str, int] = {}
    for variable in C.ALL_VARS:
        raw_values = raw[variable].to_numpy(dtype=np.float64, copy=True)
        raw_observed = raw[f"{variable}_observed"].to_numpy(copy=True)
        imputed_observed = imputed[f"{variable}_observed"].to_numpy(copy=True)
        imputed_values = imputed[variable].to_numpy(dtype=np.float64, copy=True)

        assert raw_observed.dtype == np.dtype("bool")
        assert imputed_observed.dtype == np.dtype("bool")
        np.testing.assert_array_equal(raw_observed, np.isfinite(raw_values))
        np.testing.assert_array_equal(imputed_observed, raw_observed)
        np.testing.assert_array_equal(imputed_values[raw_observed], raw_values[raw_observed])
        missing_counts[variable] = int((~raw_observed).sum())

        if variable in V5.FEATURE_VARIABLES:
            assert np.isfinite(imputed_values).all()

    # Make the preservation checks non-vacuous on both label and forcing cells.
    assert missing_counts["WTEMP"] > 0
    assert sum(missing_counts[variable] for variable in V5.METEOROLOGY_VARIABLES) > 0


def test_real_preprocessing_periods_are_exact_and_disjoint(
    production_path: _ProductionPath,
) -> None:
    raw = production_path.raw.frame
    training, validation = V5.period_masks(raw)
    dates = raw["DATE"].to_numpy(dtype="datetime64[ns]", copy=True)

    assert training.dtype == np.dtype("bool")
    assert validation.dtype == np.dtype("bool")
    assert not np.logical_and(training, validation).any()
    assert production_path.preprocessing.training_row_count == int(training.sum()) == 438_240
    assert production_path.preprocessing.validation_row_count == int(validation.sum()) == 87_720
    assert dates[training].min() == V5.TRAIN_START.to_datetime64()
    assert dates[training].max() == V5.TRAIN_END.to_datetime64()
    assert dates[validation].min() == V5.VALIDATION_START.to_datetime64()
    assert dates[validation].max() == V5.VALIDATION_END.to_datetime64()
    assert dates[training].max() < dates[validation].min()

    per_station_train = raw.loc[training].groupby("site_id", sort=True).size()
    per_station_validation = raw.loc[validation].groupby("site_id", sort=True).size()
    assert tuple(per_station_train.index) == production_path.stations
    assert tuple(per_station_validation.index) == production_path.stations
    assert per_station_train.eq(3_652).all()  # 2006-01-01 through 2015-12-31
    assert per_station_validation.eq(731).all()  # 2016-01-01 through 2017-12-31


@pytest.mark.parametrize("horizon", V5.HORIZONS)
def test_real_feature_tables_exclude_missing_labels_and_exact_bind_formal_evaluation(
    production_path: _ProductionPath,
    horizon: int,
) -> None:
    table, feature_columns = V5.build_observed_feature_table(
        production_path.raw,
        production_path.imputed,
        production_path.preprocessing.water_climatology,
        horizon,
    )
    assert feature_columns == V5.FROZEN_BASE_FEATURE_COLUMNS

    raw = production_path.raw.frame
    imputed = production_path.imputed.frame
    grid = _raw_issue_target_grid(raw, horizon)
    issue_dates = grid["issue_date"].to_numpy(dtype="datetime64[ns]", copy=True)
    target_dates = grid["target_date"].to_numpy(dtype="datetime64[ns]", copy=True)
    issue_observed = grid["issue_raw_observed"].to_numpy(dtype=np.bool_, copy=True)
    target_observed = grid["target_raw_observed"].eq(True).to_numpy(dtype=np.bool_)

    split_bounds = {
        "train": (V5.TRAIN_START, V5.TRAIN_END),
        "val": (V5.VALIDATION_START, V5.VALIDATION_END),
    }
    for split, (start, end) in split_bounds.items():
        lower = start.to_datetime64()
        upper = end.to_datetime64()
        within = (
            (issue_dates >= lower)
            & (issue_dates <= upper)
            & (target_dates >= lower)
            & (target_dates <= upper)
        )
        admissible = within & issue_observed & target_observed
        inadmissible = within & (~issue_observed | ~target_observed)
        actual = table.loc[table["split"].eq(split)].reset_index(drop=True)

        assert len(actual) == V5.EXPECTED_EXAMPLE_COUNTS[split][horizon]
        assert int((within & ~issue_observed).sum()) > 0
        assert int((within & ~target_observed).sum()) > 0
        assert _identity_index(actual).intersection(_identity_index(grid.loc[inadmissible])).empty

        expected_index = _identity_index(grid.loc[admissible])
        actual_index = _identity_index(actual)
        assert actual_index.is_unique
        assert expected_index.is_unique
        assert actual_index.difference(expected_index).empty
        assert expected_index.difference(actual_index).empty
        assert actual["issue_wtemp_observed"].all()
        assert actual["target_wtemp_observed"].all()
        assert actual["issue_date"].between(start, end).all()
        assert actual["target_date"].between(start, end).all()

        target_truth = grid.loc[
            admissible,
            [
                "site_id",
                "issue_date",
                "target_date",
                "target_raw_wtemp",
            ],
        ]
        checked_targets = actual[["site_id", "issue_date", "target_date", "y"]].merge(
            target_truth,
            on=["site_id", "issue_date", "target_date"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        np.testing.assert_array_equal(
            checked_targets["y"].to_numpy(dtype=np.float64),
            checked_targets["target_raw_wtemp"].to_numpy(dtype=np.float64),
        )

    # Lag-zero feature values come from the imputed panel, while their mask
    # channels remain the pre-imputation raw-finiteness declarations.
    fit_rows = table.loc[table["split"].isin(["train", "val"])].reset_index(drop=True)
    issue_index = pd.MultiIndex.from_frame(fit_rows[["site_id", "issue_date"]])
    truth = pd.DataFrame(index=pd.MultiIndex.from_frame(raw[["site_id", "DATE"]]))
    for variable in V5.FEATURE_VARIABLES:
        truth[f"{variable}_imputed"] = imputed[variable].to_numpy(dtype=np.float64, copy=True)
        truth[f"{variable}_raw_observed"] = raw[f"{variable}_observed"].to_numpy(
            dtype=np.bool_, copy=True
        )
    issue_truth = truth.reindex(issue_index)
    assert not issue_truth.isna().any(axis=None)
    missing_feature_cells = 0
    for variable in V5.FEATURE_VARIABLES:
        expected_mask = issue_truth[f"{variable}_raw_observed"].to_numpy(dtype=np.bool_)
        np.testing.assert_array_equal(
            fit_rows[f"{variable}_observed_lag0"].to_numpy(dtype=np.float64),
            expected_mask.astype(np.float64),
        )
        np.testing.assert_array_equal(
            fit_rows[f"{variable}_lag0"].to_numpy(dtype=np.float64),
            issue_truth[f"{variable}_imputed"].to_numpy(dtype=np.float64),
        )
        missing_feature_cells += int((~expected_mask).sum())
    assert missing_feature_cells > 0

    evaluation = V5.bind_exact_evaluation_rows(
        table,
        production_path.reference,
        horizon,
        feature_columns,
    )
    formal = (
        production_path.reference.loc[production_path.reference["lead_days"].eq(horizon)]
        .sort_values(["site_id", "issue_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        evaluation[["key_id", "site_id", "issue_date", "target_date", "y_true"]],
        formal[["key_id", "site_id", "issue_date", "target_date", "y_true"]],
        check_exact=True,
    )
    assert len(evaluation) == V5.EXPECTED_REFERENCE_COUNTS[horizon]
    assert V5.V4.target_values_equal(evaluation["y_true"], evaluation["y_panel"])

    raw_targets = raw.set_index(["site_id", "DATE"])["WTEMP"]
    evaluation_target_index = pd.MultiIndex.from_frame(
        evaluation[["site_id", "target_date"]].rename(columns={"target_date": "DATE"})
    )
    np.testing.assert_array_equal(
        evaluation["y_panel"].to_numpy(dtype=np.float64),
        raw_targets.reindex(evaluation_target_index).to_numpy(dtype=np.float64),
    )
