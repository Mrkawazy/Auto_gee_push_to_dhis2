"""Forecast and early warning: 12-month probabilistic forecasts, thresholds and rolling-origin skill."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .bases import reference_crossbasis
from .model import LEVELS, Fit, Spec
from .panel import exposure_matrix


def _period_add(period: str, months: int) -> str:
    return (pd.Period(f"{period[:4]}-{period[4:]}", freq="M") + months).strftime("%Y%m")


def climate_scenarios(cube: pd.DataFrame, climate_months: list[str], origin: str, horizon: int, mask_after_origin: bool,
                      external: pd.DataFrame | None = None, max_scenarios: int = 15) -> list[pd.DataFrame]:
    """Future climate paths for the forecast window.

    Observed values are kept up to the origin (and beyond it in live forecasts when already observed).
    Unknown months take the values from the same months k years earlier (analog years, k = 1..K),
    which keeps seasonality, month-to-month persistence and spatial coherence across facilities.
    External seasonal forecast members (e.g. ECMWF SEAS5) replace analog years where supplied.
    """
    arr = cube.to_numpy(float)
    col = {p: i for i, p in enumerate(climate_months)}
    o = col[origin]
    last = min(o + horizon, len(climate_months) - 1)
    cal = np.array([int(p[4:]) for p in climate_months])
    clim = pd.DataFrame(arr.T).groupby(cal).transform("mean").to_numpy().T  # facility calendar-month climatology
    if external is not None and len(external):
        members = sorted(external["member"].unique())
        out = []
        for m in members:
            a = arr.copy()
            ext = external.loc[external["member"] == m].pivot_table(index="facility_id", columns="period", values="value").reindex(index=cube.index)
            for c in range(o + 1, last + 1):
                p = climate_months[c]
                if mask_after_origin or np.isnan(a[:, c]).all():
                    a[:, c] = np.nan
                vals = ext[p].to_numpy(float) if p in ext.columns else np.full(len(a), np.nan)
                a[:, c] = np.where(np.isnan(a[:, c]), vals, a[:, c])
            out.append(pd.DataFrame(_fill_gaps(a, clim), index=cube.index, columns=cube.columns))
        return out
    ks = [k for k in range(1, 40) if o + 1 - 12 * k >= 0 and np.isfinite(arr[:, o + 1 - 12 * k : last + 1 - 12 * k]).mean() > 0.5][:max_scenarios]
    if not ks:
        ks = [0]
    out = []
    for k in ks:
        a = arr.copy()
        for c in range(o + 1, last + 1):
            if mask_after_origin:
                a[:, c] = np.nan
            src = a[:, c - 12 * k] if k > 0 else clim[:, c]
            a[:, c] = np.where(np.isnan(a[:, c]), src, a[:, c])
        out.append(pd.DataFrame(_fill_gaps(a, clim), index=cube.index, columns=cube.columns))
    return out


def _fill_gaps(a: np.ndarray, clim: np.ndarray) -> np.ndarray:
    """Fill any remaining gaps (including in the lag history) with climatology, then the overall median."""
    a = np.where(np.isnan(a), clim, a)
    return np.where(np.isnan(a), np.nanmedian(a) if np.isfinite(a).any() else 0.0, a)


def future_rows(hist: pd.DataFrame, facilities: pd.DataFrame, origin_period: str, origin_index: int, horizon: int, confounders: list, is_rate: bool) -> pd.DataFrame:
    """Facility x target-month rows with forecast denominators and confounders (recent facility means)."""
    rows = []
    recent = hist.loc[hist["month_index"] > origin_index - 24]
    for h in range(1, horizon + 1):
        p = _period_add(origin_period, h)
        rows.append(facilities.assign(period=p, month_index=origin_index + h, calendar_month=int(p[4:]), year=int(p[:4]), lead_months=h))
    fut = pd.concat(rows, ignore_index=True)
    if is_rate:
        by_fm = recent.groupby(["facility_id", "calendar_month"])["denominator"].mean()
        by_f = recent.groupby("facility_id")["denominator"].mean()
        den = pd.Series(list(zip(fut["facility_id"], fut["calendar_month"]))).map(by_fm.to_dict())
        fut["denominator"] = den.fillna(fut["facility_id"].map(by_f)).to_numpy()
    else:
        fut["denominator"] = np.nan
    for c in confounders:
        fut[c] = fut["facility_id"].map(recent.groupby("facility_id")[c].mean()).fillna(hist[c].mean())
    return fut


def simulate(fut: pd.DataFrame, spec: Spec, fit: Fit, scen_cbs: list[np.ndarray], n_draws: int, rng: np.random.Generator, cb_ref: np.ndarray | None):
    """Predictive draws of counts (and climate-attributable expected cases) for future rows."""
    known = fut["facility_id"].isin(spec.facilities).to_numpy()
    Xf = spec.fixed(fut)
    off = np.log(np.clip(fut["denominator"].to_numpy(float), 1e-9, None)) if spec.offset else np.zeros(len(fut))
    off = np.where(np.isfinite(off), off, 0.0)
    B = rng.multivariate_normal(fit.beta, fit.cov_model, size=n_draws, method="eigh")
    # Facility intercepts stay at their estimates: drawing them on the log scale inflates the mean
    # (exp of a wide normal) for sparse facilities, e.g. doubling rare-outcome forecasts.
    n_fe = len(spec.facilities)
    B[:, :n_fe] = fit.beta[:n_fe]
    p_fixed = Xf.shape[1]
    mu = np.zeros((len(fut), n_draws))
    an = np.zeros((len(fut), n_draws))
    scen = np.arange(n_draws) % len(scen_cbs)
    for k, cb in enumerate(scen_cbs):
        idx = np.where(scen == k)[0]
        eta = Xf @ B[idx, :p_fixed].T + off[:, None]
        if fit.p_cb:
            ce = cb @ B[idx, p_fixed:].T
            eta = eta + ce
            ae = (cb - cb_ref) @ B[idx, p_fixed:].T
        m = np.exp(np.clip(eta, -50, 30))
        mu[:, idx] = m
        if fit.p_cb:
            an[:, idx] = m * (1 - np.exp(-ae))
    mu[~known] = 0.0
    an[~known] = 0.0
    if fit.family == "negative_binomial" and fit.nb_alpha > 0:
        lam = rng.gamma(1 / fit.nb_alpha, np.maximum(mu, 1e-12) * fit.nb_alpha)
    elif fit.family == "quasi_poisson" and fit.dispersion > 1:
        lam = rng.gamma(np.maximum(mu, 1e-12) / (fit.dispersion - 1), fit.dispersion - 1)
    else:
        lam = mu
    y = rng.poisson(np.where(mu > 0, lam, 0.0)).astype(float)
    return y, mu, an


def thresholds(hist: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Alert (calendar-month P75) and epidemic (mean + 2 SD) thresholds per org unit and calendar month."""
    out = []
    for level, id_col, name_col in LEVELS:
        if level == "facility":
            series = hist[[id_col, name_col, "period", "calendar_month", "y"]].rename(columns={id_col: "org_unit_id", name_col: "org_unit_name"})
        else:
            total_fac = hist.groupby(id_col)["facility_id"].nunique()
            g = hist.groupby([id_col, name_col, "period", "calendar_month"], as_index=False).agg(y=("y", "sum"), reporting=("facility_id", "nunique"))
            g["share"] = g["reporting"] / g[id_col].map(total_fac)
            g = g.loc[g["share"] >= 0.5].copy()
            g["y"] = g["y"] / g["share"]  # scale partial months to full reporting
            series = g.rename(columns={id_col: "org_unit_id", name_col: "org_unit_name"})[["org_unit_id", "org_unit_name", "period", "calendar_month", "y"]]
        overall = series.groupby("org_unit_id")["y"].agg(["mean", "std", lambda s: np.percentile(s, settings["alert_percentile"])])
        overall.columns = ["m", "s", "p"]
        cm = series.groupby(["org_unit_id", "org_unit_name", "calendar_month"])["y"].agg(["mean", "std", "count", lambda s: np.percentile(s, settings["alert_percentile"])]).reset_index()
        cm.columns = ["org_unit_id", "org_unit_name", "calendar_month", "m", "s", "n", "p"]
        full = pd.MultiIndex.from_product([series["org_unit_id"].unique(), range(1, 13)], names=["org_unit_id", "calendar_month"]).to_frame(index=False)
        cm = full.merge(cm, on=["org_unit_id", "calendar_month"], how="left")
        cm["org_unit_name"] = cm["org_unit_id"].map(series.drop_duplicates("org_unit_id").set_index("org_unit_id")["org_unit_name"])
        small = cm["n"].fillna(0) < 3
        cm["alert_threshold"] = np.where(small, cm["org_unit_id"].map(overall["p"]), cm["p"])
        sd = np.where(small, cm["org_unit_id"].map(overall["s"]), cm["s"])
        mean = np.where(small, cm["org_unit_id"].map(overall["m"]), cm["m"])
        # Epidemic needs at least 2 events and is never below the alert threshold (rare outcomes have P75 = 0).
        cm["epidemic_threshold"] = np.maximum.reduce([mean + settings["epidemic_sd"] * np.nan_to_num(sd), cm["alert_threshold"].to_numpy(float), np.ones(len(cm))])
        cm["threshold_basis"] = np.where(small, "all months (fewer than 3 years for this calendar month)", "same calendar month")
        cm["years_of_history"] = cm["n"].fillna(0).astype(int)
        out.append(cm.assign(level=level)[["level", "org_unit_id", "org_unit_name", "calendar_month", "alert_threshold", "epidemic_threshold", "threshold_basis", "years_of_history"]])
    return pd.concat(out, ignore_index=True)


def summarise_forecast(fut: pd.DataFrame, y: np.ndarray, mu: np.ndarray, an: np.ndarray, thr: pd.DataFrame, is_rate: bool) -> pd.DataFrame:
    """Facility rows plus district/province/national sums of draws, with exceedance probabilities."""
    frames = []
    for level, id_col, name_col in LEVELS:
        codes, uniq = pd.factorize(fut[id_col].astype(str) + "|" + fut["period"])
        G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
        Y, M, A = G @ y, G @ mu, G @ an
        first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
        df = pd.DataFrame({"level": level, "org_unit_id": fut[id_col].to_numpy()[first], "org_unit_name": fut[name_col].to_numpy()[first],
                           "period": fut["period"].to_numpy()[first], "calendar_month": fut["calendar_month"].to_numpy()[first],
                           "lead_months": fut["lead_months"].to_numpy()[first]})
        df["forecast_mean"] = Y.mean(axis=1)
        q = np.quantile(Y, [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
        df["forecast_median"], df["forecast_pi80_low"], df["forecast_pi80_high"], df["forecast_pi95_low"], df["forecast_pi95_high"] = q[2], q[1], q[3], q[0], q[4]
        df["expected_cases"] = M.mean(axis=1)
        df["climate_attributable_cases"] = A.mean(axis=1)
        alo, ahi = np.quantile(A, [0.025, 0.975], axis=1)
        df["climate_attributable_cases_low"], df["climate_attributable_cases_high"] = alo, ahi
        with np.errstate(divide="ignore", invalid="ignore"):
            df["climate_attributable_fraction"] = df["climate_attributable_cases"] / df["expected_cases"]
        if is_rate:
            den = G @ np.nan_to_num(fut["denominator"].to_numpy(float))
            df["forecast_denominator"] = den
            with np.errstate(divide="ignore", invalid="ignore"):
                df["forecast_rate_per_1000"] = 1000 * df["forecast_mean"] / den
        else:
            df["forecast_denominator"] = np.nan
            df["forecast_rate_per_1000"] = np.nan
        t = thr.loc[thr["level"] == level].set_index(["org_unit_id", "calendar_month"])
        key = pd.MultiIndex.from_arrays([df["org_unit_id"], df["calendar_month"]])
        df["alert_threshold"] = t["alert_threshold"].reindex(key).to_numpy()
        df["epidemic_threshold"] = t["epidemic_threshold"].reindex(key).to_numpy()
        df["prob_exceed_alert"] = np.where(np.isfinite(df["alert_threshold"]), (Y > df["alert_threshold"].to_numpy()[:, None]).mean(axis=1), np.nan)
        df["prob_exceed_epidemic"] = np.where(np.isfinite(df["epidemic_threshold"]), (Y > df["epidemic_threshold"].to_numpy()[:, None]).mean(axis=1), np.nan)
        df["risk_level"] = np.select([df["prob_exceed_epidemic"] >= 0.5, df["prob_exceed_alert"] >= 0.5, df["prob_exceed_alert"] >= 0.25],
                                     ["high", "moderate", "watch"], "low")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def crps_samples(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Sample-based CRPS per row: E|X - y| - 0.5 E|X - X'|."""
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    term1 = np.abs(s - obs[:, None]).mean(axis=1)
    w = 2 * np.arange(1, n + 1) - n - 1
    term2 = (s * w).sum(axis=1) / (n * n)
    return term1 - term2


def score(y_draws: np.ndarray, obs: np.ndarray) -> dict:
    lo80, hi80 = np.quantile(y_draws, [0.1, 0.9], axis=1)
    lo95, hi95 = np.quantile(y_draws, [0.025, 0.975], axis=1)
    mean = y_draws.mean(axis=1)
    return {"crps": crps_samples(y_draws, obs), "abs_error": np.abs(mean - obs), "error": mean - obs,
            "cover80": (obs >= lo80) & (obs <= hi80), "cover95": (obs >= lo95) & (obs <= hi95)}
