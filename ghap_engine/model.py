"""Analytics layer: DLNM fitting, attribution with Monte Carlo uncertainty, diagnostics and QA."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

from .bases import NaturalSpline, crossbasis, cumulative_gradient, fourier, lag_basis, lag_gradient, reference_crossbasis
from .panel import exposure_matrix

ER_PERCENTILES = [1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97.5, 99]
BLOCKING_FLAGS = {"fit_failed", "non_finite_estimates", "implausible_effect", "uninformative_ci", "rank_deficient", "mapping_warning"}
COMPONENTS = ["above_ref", "below_ref", "extreme_high", "extreme_low"]


@dataclass
class Spec:
    """Everything needed to rebuild the design matrix for new rows (fitting, forecasting, CV)."""

    facilities: list
    harmonics: int
    trend: NaturalSpline
    trend_max: float
    events: dict
    confounders: list
    exposure: NaturalSpline | None
    cmat: np.ndarray | None
    lag_max: int
    offset: bool
    keep: np.ndarray | None = None  # boolean mask of fixed columns retained after dropping constants

    def fixed(self, rows: pd.DataFrame) -> np.ndarray:
        codes = pd.Categorical(rows["facility_id"], categories=self.facilities).codes
        fe = np.zeros((len(rows), len(self.facilities)))
        known = codes >= 0
        fe[np.nonzero(known)[0], codes[known]] = 1.0
        t = rows["month_index"].to_numpy(float)
        tb = self.trend(np.minimum(t, self.trend_max))
        future = t > self.trend_max
        if future.any():  # forecasts hold the trend at its mean over the last 12 fitted months (end points are noisy)
            tb[future] = self.trend(np.arange(self.trend_max - 11, self.trend_max + 1)).mean(axis=0)
        parts = [fe, fourier(rows["calendar_month"].to_numpy(), self.harmonics), tb]
        for _, (s, e) in self.events.items():
            parts.append(((rows["period"] >= s) & (rows["period"] <= e)).to_numpy(float)[:, None])
        for c in self.confounders:
            parts.append(rows[c].to_numpy(float)[:, None])
        X = np.column_stack(parts)
        return X if self.keep is None else X[:, self.keep]

    def cb(self, lagged: np.ndarray) -> np.ndarray:
        return crossbasis(lagged, self.exposure, self.cmat)


@dataclass
class Fit:
    beta: np.ndarray
    cov_robust: np.ndarray
    cov_model: np.ndarray
    family: str
    dispersion: float
    nb_alpha: float
    cov_type: str
    mu: np.ndarray
    pearson: np.ndarray
    llf: float
    qaic: float
    pseudo_r2: float
    n_params: int
    p_cb: int
    extra: dict = field(default_factory=dict)

    @property
    def beta_cb(self):
        return self.beta[-self.p_cb :] if self.p_cb else np.array([])

    @property
    def cov_cb(self):
        return self.cov_robust[-self.p_cb :, -self.p_cb :] if self.p_cb else np.empty((0, 0))


def variance(mu: np.ndarray, family: str, dispersion: float, alpha: float) -> np.ndarray:
    if family == "negative_binomial":
        return mu + alpha * mu**2
    if family == "quasi_poisson":
        return dispersion * mu
    return mu


def build_spec(rows: pd.DataFrame, settings: dict, exposure_values: np.ndarray | None, lag_max: int, offset: bool, confounders: list, exposure_df: int | None = None) -> Spec:
    n_months = rows["month_index"].nunique()
    # One trend df per `trend_years_per_df` years, but no more than one per `trend_events_per_df` events (rare outcomes get a linear trend).
    trend_df = max(1, min(int(round(n_months / 12 / settings["trend_years_per_df"])), int(rows["y"].sum() // settings["trend_events_per_df"])))
    t = rows["month_index"].to_numpy(float)
    events = {}
    for name, (s, e) in settings.get("event_periods", {}).items():
        s, e = s.replace("-", ""), e.replace("-", "")
        inside = (rows["period"] >= s) & (rows["period"] <= e)
        if 0 < inside.sum() < len(rows):  # only identifiable disruptions enter the model
            events[name] = (s, e)
    exposure = cmat = None
    if exposure_values is not None:
        vals = exposure_values[np.isfinite(exposure_values)]
        df = exposure_df or settings["exposure_df"]
        if len(np.unique(vals)) < 10:
            df = 1  # near-discrete exposures (e.g. heatwave counts) get a linear term
        exposure = NaturalSpline.from_data(vals, df)
        cmat = lag_basis(lag_max)
    facilities = sorted(rows.loc[rows.groupby("facility_id")["y"].transform("sum") > 0, "facility_id"].unique())
    return Spec(facilities, settings["fourier_harmonics"], NaturalSpline.from_data(t, trend_df), float(t.max()), events, confounders, exposure, cmat, lag_max, offset)


def _xlogy(y, x):
    return np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / np.where(x > 0, x, 1.0)), 0.0)


def deviance(y, mu, alpha=0.0) -> float:
    if alpha > 0:
        return float(2 * np.sum(_xlogy(y, mu) - (y + 1 / alpha) * np.log((1 + alpha * y) / (1 + alpha * mu))))
    return float(2 * np.sum(_xlogy(y, mu) - (y - mu)))


def loglik(y, mu, alpha=0.0) -> float:
    if alpha > 0:
        r = 1 / alpha
        return float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + y * np.log(alpha * mu / (1 + alpha * mu)) - r * np.log1p(alpha * mu)))
    return float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1)))


def irls(y, X, offset, alpha=0.0, beta=None, maxiter=100, tol=1e-10):
    """Log-link Poisson (alpha=0) or NB2 (fixed alpha) by IRLS with Cholesky solves and step halving.

    Returns (beta, mu, information matrix X'WX at the solution, converged).
    """
    if beta is None:
        mu = (y + y.mean()) / 2 + 1e-3
        eta = np.log(mu)
        w = mu / (1 + alpha * mu)
        XtW = X.T * w
        beta = cho_solve(cho_factor(XtW @ X), XtW @ (eta - offset + (y - mu) / mu))
    eta = X @ beta + offset
    mu = np.exp(np.clip(eta, -50, 50))
    dev = deviance(y, mu, alpha)
    converged = False
    for _ in range(maxiter):
        w = mu / (1 + alpha * mu)
        XtW = X.T * w
        new = cho_solve(cho_factor(XtW @ X), XtW @ (eta - offset + (y - mu) / mu))
        step = new - beta
        for _ in range(30):  # step halving keeps the deviance non-increasing
            cand = beta + step
            eta_c = X @ cand + offset
            mu_c = np.exp(np.clip(eta_c, -50, 50))
            dev_c = deviance(y, mu_c, alpha)
            if np.isfinite(dev_c) and dev_c <= dev + 1e-12 * (abs(dev) + 1):
                break
            step /= 2
        beta, eta, mu = cand, eta_c, mu_c
        if abs(dev - dev_c) <= tol * (abs(dev_c) + 0.1):
            dev = dev_c
            converged = True
            break
        dev = dev_c
    w = mu / (1 + alpha * mu)
    return beta, mu, (X.T * w) @ X, converged


def cluster_sandwich(X, y, mu, alpha, info, groups) -> np.ndarray:
    """Facility-clustered sandwich covariance (CR1 small-sample correction, as statsmodels)."""
    bread = np.linalg.inv(info)
    scores = X * ((y - mu) / (1 + alpha * mu))[:, None]
    codes, uniq = pd.factorize(groups)
    G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    sg = G @ scores
    meat = sg.T @ sg
    n, k, g = X.shape[0], X.shape[1], len(uniq)
    corr = g / (g - 1) * (n - 1) / (n - k)
    return corr * bread @ meat @ bread


def fit_glm(y, X, offset, groups, settings, p_cb: int, family_override: tuple[str, float] | None = None) -> Fit:
    """Poisson fit, then quasi-Poisson or negative binomial by the Cameron-Trivedi variance check (framework section 10).

    `family_override=(family, nb_alpha)` reuses a family chosen on the full data (district and CV refits).
    """
    beta, mu, info, conv = irls(y, X, offset)
    pois_dev, pois_llf = deviance(y, mu), loglik(y, mu)
    dfr = max(1, len(y) - X.shape[1])
    phi = float(np.sum((y - mu) ** 2 / mu) / dfr)
    family, alpha = "poisson", 0.0
    if family_override is not None:
        family, alpha = family_override
        if family == "negative_binomial":
            beta, mu, info, conv = irls(y, X, offset, alpha=alpha, beta=beta)
    elif phi > settings["overdispersion_threshold"]:
        z = ((y - mu) ** 2 - y) / mu
        a = float(np.sum(z * mu) / np.sum(mu * mu))
        rss_nb2 = np.sum((z - a * mu) ** 2)
        rss_nb1 = np.sum((z - z.mean()) ** 2)
        if a > 0 and rss_nb2 < rss_nb1:  # excess variance grows with the mean -> NB2
            family = "negative_binomial"
            for _ in range(2):  # re-estimate alpha (method of moments) at the NB fit
                beta, mu, info, conv = irls(y, X, offset, alpha=a, beta=beta)
                a = max(1e-6, float(np.sum(((y - mu) ** 2 - mu) / mu**2) / dfr))
            alpha = a
            beta, mu, info, conv = irls(y, X, offset, alpha=alpha, beta=beta)
        else:
            family = "quasi_poisson"
    if not conv:
        warnings.warn("IRLS did not fully converge")
    var = variance(mu, family, phi, alpha)
    pearson = (y - mu) / np.sqrt(var)
    disp = phi if family != "negative_binomial" else float(np.sum(pearson**2) / dfr)
    cov_model = np.linalg.inv(info) * (phi if family == "quasi_poisson" else 1.0)
    if len(np.unique(groups)) >= settings["cluster_min_groups"]:
        cov_robust, cov_type = cluster_sandwich(X, y, mu, alpha, info, groups), "cluster_robust_facility"
    else:
        cov_robust, cov_type = cov_model, f"model_based_{family}"
    llf = pois_llf if family != "negative_binomial" else loglik(y, mu, alpha)
    k = X.shape[1] + (1 if family != "poisson" else 0)
    qaic = -2 * llf / (phi if family == "quasi_poisson" else 1.0) + 2 * k
    mu0 = np.exp(offset) * y.sum() / np.exp(offset).sum()
    null_dev = deviance(y, mu0)
    pseudo_r2 = 1 - pois_dev / null_dev if null_dev > 0 else np.nan
    return Fit(beta, cov_robust, cov_model, family, disp, alpha, cov_type, mu, pearson, llf, qaic, pseudo_r2, X.shape[1], p_cb, {"converged": conv})


def full_rank(X: np.ndarray) -> bool:
    xtx = X.T @ X
    d = np.sqrt(np.clip(np.diag(xtx), 1e-300, None))
    ev = np.linalg.eigvalsh(xtx / np.outer(d, d))
    return bool(ev.min() > 1e-10 * ev.max())


def drop_constant_columns(Xf: np.ndarray, n_fe: int) -> np.ndarray:
    keep = np.ones(Xf.shape[1], bool)
    sd = Xf[:, n_fe:].std(axis=0)
    keep[n_fe:] = sd > 1e-10
    return keep


def fit_pair_model(rows: pd.DataFrame, lagged: np.ndarray, settings: dict, lag_max: int, offset: bool, confounders: list,
                   spec: Spec | None = None, family_override: tuple[str, float] | None = None) -> tuple[Spec, Fit, np.ndarray]:
    """Fit the DLNM on rows with events (facility FE). Returns (spec, fit, row mask used in the fit)."""
    if spec is None:
        spec = build_spec(rows, settings, lagged.ravel() if lagged is not None else None, lag_max, offset, confounders)
    use = rows["facility_id"].isin(spec.facilities).to_numpy()
    r = rows.loc[use]
    spec.keep = None
    Xf = spec.fixed(r)
    spec.keep = drop_constant_columns(Xf, len(spec.facilities))
    Xf = Xf[:, spec.keep]
    cb = spec.cb(lagged[use]) if spec.exposure is not None else np.empty((len(r), 0))
    X = np.column_stack([Xf, cb])
    if not full_rank(X):
        raise np.linalg.LinAlgError("rank_deficient")
    y = r["y"].to_numpy(float)
    off = np.log(r["denominator"].to_numpy(float)) if offset else np.zeros(len(r))
    fit = fit_glm(y, X, off, r["facility_id"].to_numpy(), settings, cb.shape[1], family_override)
    return spec, fit, use


# ---------------------------------------------------------------------------------
# Effect summaries
# ---------------------------------------------------------------------------------


def log_rr_ci(g: np.ndarray, beta: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est = g @ beta
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", g, cov, g), 0, None))
    return est, se


def find_mmt(spec: Spec, fit: Fit, pooled: np.ndarray, settings: dict) -> tuple[float, float]:
    """Minimum-risk exposure on the pooled cumulative curve, searched within the configured percentile range."""
    lo, hi = settings["mmt_search_percentiles"]
    pct = np.arange(lo, hi + 0.5, 0.5)
    grid = np.percentile(pooled, pct)
    est = cumulative_gradient(grid, float(np.median(pooled)), spec.exposure, spec.cmat) @ fit.beta_cb
    i = int(np.argmin(est))
    return float(grid[i]), float(pct[i])


def reference_value(kind: str, pooled: np.ndarray, mmt: float) -> float:
    return {"mmt": mmt, "median": float(np.median(pooled)), "p25": float(np.percentile(pooled, 25)), "p75": float(np.percentile(pooled, 75))}[kind]


def exposure_response_table(spec, fit, pooled, refs: dict) -> pd.DataFrame:
    grid = np.percentile(pooled, ER_PERCENTILES)
    out = []
    for ref_kind, ref in refs.items():
        g = cumulative_gradient(grid, ref, spec.exposure, spec.cmat)
        est, se = log_rr_ci(g, fit.beta_cb, fit.cov_cb)
        out.append(pd.DataFrame({"reference_type": ref_kind, "reference_value": ref, "exposure_percentile": ER_PERCENTILES, "exposure_value": grid,
                                 "rr": np.exp(est), "rr_low": np.exp(est - 1.96 * se), "rr_high": np.exp(est + 1.96 * se)}))
    return pd.concat(out, ignore_index=True)


def lag_response_table(spec, fit, pooled, ref) -> pd.DataFrame:
    out = []
    for p in [1, 10, 90, 99]:
        x = float(np.percentile(pooled, p))
        g = np.array([lag_gradient(x, ref, lag, spec.exposure, spec.cmat) for lag in range(spec.lag_max + 1)])
        est, se = log_rr_ci(g, fit.beta_cb, fit.cov_cb)
        out.append(pd.DataFrame({"exposure_percentile": p, "exposure_value": x, "lag": np.arange(spec.lag_max + 1),
                                 "rr": np.exp(est), "rr_low": np.exp(est - 1.96 * se), "rr_high": np.exp(est + 1.96 * se)}))
    return pd.concat(out, ignore_index=True)


def draw_cb(fit: Fit, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(fit.beta_cb, fit.cov_cb, size=n, method="eigh")


# ---------------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------------


def attribution(rows: pd.DataFrame, lagged: np.ndarray, spec: Spec, fit: Fit, ref: float, cuts: tuple[float, float], draws: np.ndarray) -> dict:
    """Backward attributable fractions/numbers per row, with draw matrices for aggregation.

    AF_t = 1 - exp(-(cb_t - cb_ref) beta): share of cases in month t due to the exposure
    history (lags 0..L) differing from the reference. AN_t = AF_t * observed_t.
    Components re-evaluate the cross-basis with exposures outside the range set to the reference.
    """
    y = rows["y"].to_numpy(float)
    cb_ref = reference_crossbasis(ref, spec.exposure, spec.cmat)
    # Extreme components never cross the reference: extreme heat is above max(P90, ref), extreme cold below min(P10, ref).
    ranges = {"total": None, "above_ref": (ref, np.inf), "below_ref": (-np.inf, ref),
              "extreme_high": (max(cuts[1], ref), np.inf), "extreme_low": (-np.inf, min(cuts[0], ref))}
    draws32 = draws.astype(np.float32)
    y32 = y.astype(np.float32)[:, None]
    out = {}
    for name, bounds in ranges.items():
        lag_x = lagged if bounds is None else np.where((lagged > bounds[0]) & (lagged < bounds[1]), lagged, ref)
        diff = spec.cb(lag_x) - cb_ref
        eta = diff @ fit.beta_cb
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, fit.cov_cb, diff), 0, None))
        af_s = 1 - np.exp(-(diff.astype(np.float32) @ draws32.T))
        out[name] = {"af": 1 - np.exp(-eta), "an": (1 - np.exp(-eta)) * y, "an_draws": af_s * y32, "eta": eta, "se": se}
    return out


def row_table(rows: pd.DataFrame, lagged: np.ndarray, att: dict, mu: np.ndarray, ref: float) -> pd.DataFrame:
    # Row-level 95% limits: monotone transforms of the normal interval of eta (exact under the normal approximation).
    tot = att["total"]
    y = rows["y"].to_numpy(float)
    lo_eta, hi_eta = tot["eta"] - 1.96 * tot["se"], tot["eta"] + 1.96 * tot["se"]
    rr_lo, rr_hi = np.exp(lo_eta), np.exp(hi_eta)
    af_lo, af_hi = 1 - np.exp(-lo_eta), 1 - np.exp(-hi_eta)
    an_lo, an_hi = af_lo * y, af_hi * y
    df = rows[["facility_id", "facility_name", "district_id", "district_name", "province_id", "province_name", "period", "year"]].copy()
    df["observed"] = rows["y"].to_numpy()
    df["denominator"] = rows["denominator"].to_numpy()
    df["expected"] = mu
    df["exposure_lag0"] = lagged[:, 0]
    df["exposure_window_mean"] = lagged.mean(axis=1)
    df["reference_exposure"] = ref
    df["rr"] = np.exp(tot["eta"])
    df["rr_low"], df["rr_high"] = rr_lo, rr_hi
    df["af"], df["af_low"], df["af_high"] = tot["af"], af_lo, af_hi
    df["an"], df["an_low"], df["an_high"] = tot["an"], an_lo, an_hi
    for c in COMPONENTS:
        e, s = att[c]["eta"], att[c]["se"]
        df[f"an_{c}"], df[f"an_{c}_low"], df[f"an_{c}_high"] = att[c]["an"], (1 - np.exp(-(e - 1.96 * s))) * y, (1 - np.exp(-(e + 1.96 * s))) * y
    df["expected_without_exposure"] = mu * np.exp(-tot["eta"])
    return df


LEVELS = [("facility", "facility_id", "facility_name"), ("district", "district_id", "district_name"),
          ("province", "province_id", "province_name"), ("national", "national_id", "national_name")]


def aggregate_attribution(rows: pd.DataFrame, att: dict, mu: np.ndarray, is_rate: bool) -> pd.DataFrame:
    """Sum attributable numbers within each draw over org units and periods, then take quantiles.

    This yields valid uncertainty for district/province/national monthly, annual and whole-period totals
    (ratios and interval bounds must never be averaged across units).
    """
    comps = ["total"] + COMPONENTS
    base = {"facility_months": np.ones(len(rows)), "observed": rows["y"].to_numpy(float), "expected": mu,
            "denominator": np.nan_to_num(rows["denominator"].to_numpy(float))}
    points = {c: att[c]["an"] for c in comps}
    draws = {c: att[c]["an_draws"] for c in comps}

    def group(keys: pd.Series):
        codes, uniq = pd.factorize(keys)
        G = sparse.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
        first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
        return G, first

    def apply(G, b, p, d):
        return {k: G @ v for k, v in b.items()}, {k: G @ v for k, v in p.items()}, {k: G @ v for k, v in d.items()}

    def emit(level, ptype, meta, b, p, d):
        agg = meta.assign(level=level, period_type=ptype)
        for k, v in b.items():
            agg[k] = v
        if not is_rate:
            agg["denominator"] = np.nan
        obs, dsum = agg["observed"].to_numpy(), agg["denominator"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            for c in comps:
                pre = "an" if c == "total" else f"an_{c}"
                lo, hi = np.quantile(d[c], [0.025, 0.975], axis=1)
                agg[pre], agg[f"{pre}_low"], agg[f"{pre}_high"] = p[c], lo, hi
            agg["paf"] = p["total"] / obs
            agg["paf_low"], agg["paf_high"] = np.quantile(d["total"] / obs[:, None], [0.025, 0.975], axis=1)
            if is_rate:
                agg["attributable_rate_per_1000"] = 1000 * p["total"] / dsum
                agg["attributable_rate_per_1000_low"], agg["attributable_rate_per_1000_high"] = np.quantile(1000 * d["total"] / dsum[:, None], [0.025, 0.975], axis=1)
            else:
                agg["attributable_rate_per_1000"] = agg["attributable_rate_per_1000_low"] = agg["attributable_rate_per_1000_high"] = np.nan
        frames.append(agg)

    # Hierarchical sums within each draw: rows -> org-unit month (or facility-year) -> year -> whole period.
    frames = []
    for level, id_col, name_col in LEVELS:
        stage = "year" if level == "facility" else "month"
        per = rows["year"].astype(str) if stage == "year" else rows["period"]
        G1, first = group(rows[id_col].astype(str) + "|" + per)
        meta1 = pd.DataFrame({"org_unit_id": rows[id_col].to_numpy()[first], "org_unit_name": rows[name_col].to_numpy()[first],
                              "period": per.to_numpy()[first], "_year": rows["year"].astype(str).to_numpy()[first]})
        b1, p1, d1 = apply(G1, base, points, draws)
        emit(level, stage, meta1[["org_unit_id", "org_unit_name", "period"]], b1, p1, d1)
        if stage == "month":
            G2, f2 = group(meta1["org_unit_id"].astype(str) + "|" + meta1["_year"])
            emit(level, "year", meta1.iloc[f2][["org_unit_id", "org_unit_name"]].assign(period=meta1["_year"].to_numpy()[f2]).reset_index(drop=True), *apply(G2, b1, p1, d1))
        G3, f3 = group(meta1["org_unit_id"].astype(str))
        emit(level, "total", meta1.iloc[f3][["org_unit_id", "org_unit_name"]].assign(period="all").reset_index(drop=True), *apply(G3, b1, p1, d1))
    cols = ["level", "org_unit_id", "org_unit_name", "period_type", "period"]
    out = pd.concat(frames, ignore_index=True)
    return out[cols + [c for c in out.columns if c not in cols]]


# ---------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------


def residual_autocorrelation(rows: pd.DataFrame, resid: np.ndarray) -> tuple[float, float]:
    s = pd.Series(resid, index=pd.MultiIndex.from_arrays([rows["facility_id"], rows["month_index"]]))
    out = []
    for lag in (1, 12):
        shifted = pd.Series(s.values, index=pd.MultiIndex.from_arrays([rows["facility_id"], rows["month_index"] + lag]))
        joined = pd.concat([s.rename("a"), shifted.rename("b")], axis=1, join="inner").dropna()
        out.append(float(np.corrcoef(joined["a"], joined["b"])[0, 1]) if len(joined) > 10 else np.nan)
    return out[0], out[1]


def residual_seasonality_p(rows: pd.DataFrame, resid: np.ndarray) -> float:
    groups = [resid[(rows["calendar_month"] == m).to_numpy()] for m in range(1, 13)]
    groups = [g for g in groups if len(g) > 1]
    return float(stats.f_oneway(*groups).pvalue) if len(groups) > 2 else np.nan


def morans_i(fac: pd.DataFrame, rows: pd.DataFrame, resid: np.ndarray, rng: np.random.Generator, perms: int = 499) -> tuple[float, float]:
    m = pd.Series(resid).groupby(rows["facility_id"].to_numpy()).mean()
    f = fac.set_index("facility_id").reindex(m.index)
    ok = f["lon"].notna().to_numpy() & f["lat"].notna().to_numpy()
    if ok.sum() < 5:
        return np.nan, np.nan
    v = m.to_numpy()[ok]
    lon, lat = np.radians(f["lon"].to_numpy()[ok]), np.radians(f["lat"].to_numpy()[ok])
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    d = 2 * 6371 * np.arcsin(np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2))
    with np.errstate(divide="ignore"):
        w = np.where(d > 0, 1 / np.maximum(d, 1e-6), 0.0)
    w = w / w.sum(axis=1, keepdims=True)

    def stat(x):
        z = x - x.mean()
        return len(z) * (z @ w @ z) / (z @ z) / w.sum() if z @ z > 0 else 0.0

    obs = stat(v)
    null = np.array([stat(rng.permutation(v)) for _ in range(perms)])
    return float(obs), float((np.sum(null >= obs) + 1) / (perms + 1))


def dersimonian_laird(est: np.ndarray, var: np.ndarray) -> dict:
    w = 1 / var
    mu_fe = np.sum(w * est) / np.sum(w)
    q = float(np.sum(w * (est - mu_fe) ** 2))
    dfree = len(est) - 1
    tau2 = max(0.0, (q - dfree) / (np.sum(w) - np.sum(w**2) / np.sum(w))) if dfree > 0 else 0.0
    wr = 1 / (var + tau2)
    mu = np.sum(wr * est) / np.sum(wr)
    shrink = tau2 / (tau2 + var)
    blup = mu + shrink * (est - mu)
    blup_var = shrink * var + (1 - shrink) ** 2 / np.sum(wr)
    i2 = max(0.0, (q - dfree) / q) if q > 0 else 0.0
    return {"pooled": mu, "pooled_se": np.sqrt(1 / np.sum(wr)), "q": q, "df": dfree, "q_p": float(stats.chi2.sf(q, dfree)) if dfree > 0 else np.nan,
            "i2": i2, "tau2": tau2, "blup": blup, "blup_se": np.sqrt(blup_var)}
