"""Baseline forecasters, from the unbeatable-looking persistence floor up to a
half-physical thermal-relaxation model and LightGBM.

Each ``run_*`` returns rows in the canonical predictions schema (``results.py``)
so baselines and ThermoRoute are scored by exactly the same code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from . import config as C
from . import features as F
from . import results as R


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tab_by_horizon(panel, clim, variables):
    return {h: F.attach_split(F.build_tabular(panel, h, variables, clim))
            for h in C.HORIZONS}


def _base_cols(tab: pd.DataFrame, h: int, model: str, y_pred: np.ndarray,
               feature_set: str = "-", scope: str = "per_station") -> pd.DataFrame:
    return R.make_pred_frame(
        model=model, scope=scope, feature_set=feature_set, seed=0,
        site_id=tab["site_id"].to_numpy(), horizon=np.full(len(tab), h),
        split=tab["split"].to_numpy(), issue_date=tab["issue_date"].to_numpy(),
        target_date=tab["target_date"].to_numpy(),
        y_true=tab["y"].to_numpy(float), y_pred=y_pred,
    )


# --------------------------------------------------------------------------- #
# Trivial / seasonal baselines
# --------------------------------------------------------------------------- #
def run_persistence(tabs) -> pd.DataFrame:
    out = [_base_cols(tab, h, "Persistence", tab["persistence"].to_numpy(float))
           for h, tab in tabs.items()]
    return pd.concat(out, ignore_index=True)


def run_climatology(tabs) -> pd.DataFrame:
    out = [_base_cols(tab, h, "Climatology", tab["clim_target"].to_numpy(float))
           for h, tab in tabs.items()]
    return pd.concat(out, ignore_index=True)


def run_damped_persistence(panel, masks, tabs) -> pd.DataFrame:
    """ŷ_{t+h} = clim_{t+h} + φ_s^h (y_t − clim_t).  φ_s = train AR(1) of anomaly."""
    phi = {}
    # estimate φ from the tabular anomaly (y_t − clim_t) on the training split
    base = tabs[1]
    for st in C.STATIONS:
        a = base[(base.site_id == st) & (base.split == "train")]["clim_anom"].to_numpy(float)
        phi[st] = float(np.clip(np.corrcoef(a[1:], a[:-1])[0, 1], 0.0, 0.999))
    out = []
    for h, tab in tabs.items():
        ph = tab["site_id"].map(phi).to_numpy(float) ** h
        yhat = tab["clim_target"].to_numpy(float) + ph * (
            tab["persistence"].to_numpy(float) - tab["clim_t"].to_numpy(float))
        out.append(_base_cols(tab, h, "DampedPersistence", yhat))
    return pd.concat(out, ignore_index=True), phi


# --------------------------------------------------------------------------- #
# Ridge (regularised linear / GAM-style statistical baseline)
# --------------------------------------------------------------------------- #
def run_ridge(tabs, feature_set: str = "V3") -> pd.DataFrame:
    out = []
    for h, tab in tabs.items():
        cols = F.feature_columns(tab)
        for st in C.STATIONS:
            sub = tab[tab.site_id == st]
            tr = sub[sub.split == "train"]
            sc = StandardScaler().fit(tr[cols].to_numpy(float))
            model = Ridge(alpha=10.0).fit(sc.transform(tr[cols].to_numpy(float)),
                                          tr["y"].to_numpy(float))
            yhat = model.predict(sc.transform(sub[cols].to_numpy(float)))
            out.append(_base_cols(sub, h, "Ridge", yhat, feature_set=feature_set))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# air2stream-lite: half-physical thermal relaxation (constant rate, flow-modulated)
# --------------------------------------------------------------------------- #
def _fit_air2stream(W, TEMP, logQz):
    """Calibrate {a,b,k0,k1} on 1-step-ahead MSE.

    W_{t+1} = W_t + k_t (Teq_t − W_t),  Teq = a + b·TEMP,
    k_t = sigmoid(k0 + k1·z(logFLOW)) ∈ (0,1).
    """
    def resid(p):
        a, b, k0, k1 = p
        k = 1.0 / (1.0 + np.exp(-(k0 + k1 * logQz[:-1])))
        teq = a + b * TEMP[:-1]
        pred = W[:-1] + k * (teq - W[:-1])
        return pred - W[1:]
    p0 = np.array([W.mean(), 0.3, 0.0, 0.0])
    sol = least_squares(resid, p0, max_nfev=4000)
    return sol.x


def run_air2stream(panel, masks, clim_air) -> pd.DataFrame:
    """Roll the calibrated relaxation forward h steps using climatological air
    temperature for the future (a fair Track-H half-physical baseline)."""
    out = []
    tr_mask = masks.train
    for st in C.STATIONS:
        sub = panel[panel.site_id == st].sort_values("DATE").reset_index(drop=True)
        W = sub[C.TARGET].to_numpy(float)
        TEMP = sub["TEMP"].to_numpy(float)
        logQ = np.log1p(sub["FLOW"].to_numpy(float))
        tr_rows = tr_mask[(panel.site_id == st).to_numpy()]
        qmu, qsd = logQ[tr_rows].mean(), logQ[tr_rows].std() + 1e-8
        logQz = (logQ - qmu) / qsd
        a, b, k0, k1 = _fit_air2stream(W[tr_rows], TEMP[tr_rows], logQz[tr_rows])

        doy = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
        temp_clim = clim_air.predict(st, doy)  # seasonal air-temp expectation
        n = len(sub)
        for h in C.HORIZONS:
            yhat = np.full(n, np.nan)
            for t in range(n - h):
                w = W[t]
                kz = logQz[t]  # persist flow regime over the short horizon
                k = 1.0 / (1.0 + np.exp(-(k0 + k1 * kz)))
                for i in range(1, h + 1):
                    teq = a + b * temp_clim[t + i]
                    w = w + k * (teq - w)
                yhat[t] = w
            tabish = pd.DataFrame({
                "site_id": st, "issue_date": sub["DATE"].to_numpy(),
                "target_date": (sub["DATE"] + pd.to_timedelta(h, "D")).to_numpy(),
                "persistence": W, "y": np.r_[W[h:], np.full(h, np.nan)],
            })
            tabish["split"] = F.attach_split(tabish.rename(columns={}))["split"]
            valid = ~np.isnan(yhat) & ~np.isnan(tabish["y"].to_numpy(float))
            tabish = tabish[valid]
            out.append(_base_cols(tabish, h, "Air2streamLite",
                                  yhat[valid], feature_set="phys"))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# LightGBM: strong ML baseline (point + quantiles + exceedance probability)
# --------------------------------------------------------------------------- #
def _lgb_fit(Xtr, ytr, Xval, yval, objective, alpha=None, n_est=800):
    # n_jobs=1 avoids an OpenMP (libomp/libiomp) conflict with PyTorch that
    # segfaults when both are imported in the same process on macOS/anaconda.
    params = dict(objective=objective, learning_rate=0.03, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, n_estimators=n_est,
                  verbosity=-1, seed=0, n_jobs=1)
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m


def run_lightgbm(tabs, thresholds, feature_set: str = "V3",
                 quantiles=True) -> pd.DataFrame:
    out = []
    for h, tab in tabs.items():
        cols = F.feature_columns(tab)
        for st in C.STATIONS:
            sub = tab[tab.site_id == st]
            tr, va = sub[sub.split == "train"], sub[sub.split == "val"]
            Xtr, ytr = tr[cols].to_numpy(float), tr["y"].to_numpy(float)
            Xva, yva = va[cols].to_numpy(float), va["y"].to_numpy(float)
            Xall = sub[cols].to_numpy(float)

            mp = _lgb_fit(Xtr, ytr, Xva, yva, "regression")
            yhat = mp.predict(Xall)
            frame = _base_cols(sub, h, "LightGBM", yhat, feature_set=feature_set)

            if quantiles:
                preds = {}
                for q in C.QUANTILES:
                    mq = _lgb_fit(Xtr, ytr, Xva, yva, "quantile", alpha=q)
                    preds[q] = mq.predict(Xall)
                # enforce monotonicity
                stacked = np.sort(np.vstack([preds[0.05], preds[0.50], preds[0.95]]), axis=0)
                frame["q05"], frame["q50"], frame["q95"] = stacked[0], stacked[1], stacked[2]
                # exceedance probability from a binary classifier
                thr = thresholds[st]
                clf = lgb.LGBMClassifier(
                    n_estimators=600, learning_rate=0.03, num_leaves=31,
                    min_child_samples=40, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, seed=0, n_jobs=1)
                clf.fit(Xtr, (ytr > thr).astype(int), eval_set=[(Xva, (yva > thr).astype(int))],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                frame["p_exceed"] = clf.predict_proba(Xall)[:, 1]
            out.append(frame)
    return pd.concat(out, ignore_index=True)
