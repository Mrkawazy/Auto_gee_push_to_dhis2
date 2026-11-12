"""Processing layer: validation, zero-fill of submitted forms, tail trimming and analysis panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import analysis_end_month

REASONS = {
    "numerator_not_reported": "A numerator component was blank and the form was not submitted",
    "denominator_not_reported": "A denominator component was blank and the form was not submitted",
    "denominator_below_minimum": "Mapped denominator below the minimum (or zero)",
    "ratio_above_max": "Numerator exceeds the denominator (data error)",
    "missing_confounder": "A configured confounder was not reported",
    "incomplete_recent_reporting": "Recent month with incomplete reporting (late reports)",
    "missing_exposure_history": "Climate value missing for a month in the lag window",
}


def _period_str(p: pd.Period) -> str:
    return p.strftime("%Y%m")


def build_panels(inputs: dict, settings: dict) -> dict:
    """Return outcome panels, climate cubes and data-quality tables."""
    fac = inputs["facilities"]
    fac_ids = fac["facility_id"].tolist()
    start = pd.Period(settings["start_month"], freq="M")
    end = analysis_end_month(settings)
    months = pd.period_range(start, end, freq="M")
    pstr = [_period_str(p) for p in months]
    values = inputs["values"]
    health = values.loc[values["kind"] == "health"].pivot_table(index=["ou", "period"], columns="dx", values="value", aggfunc="sum")
    grid = pd.MultiIndex.from_product([fac_ids, pstr], names=["ou", "period"])
    health = health.reindex(grid)

    reports = inputs["reports"]
    submitted = (
        reports.loc[reports["value"] > 0].assign(flag=True).pivot_table(index=["ou", "period"], columns="dataset", values="flag", aggfunc="max")
        if not reports.empty else pd.DataFrame(index=grid)
    )
    submitted = submitted.reindex(grid).fillna(False).astype(bool)

    def operand_values(op: str) -> np.ndarray:
        vals = health[op].to_numpy(float) if op in health.columns else np.full(len(grid), np.nan)
        if settings["zero_fill_when_form_submitted"]:
            ds = [d for d in inputs["operand_datasets"].get(op.split(".")[0], []) if d in submitted.columns]
            if ds:
                form = submitted[ds].any(axis=1).to_numpy()
                vals = np.where(np.isnan(vals) & form, 0.0, vals)
        return vals

    def expression(ops: list[str]) -> np.ndarray:
        return np.sum([operand_values(op) for op in ops], axis=0) if ops else np.full(len(grid), np.nan)

    conf = {}
    from .inputs import parse_expression

    for name, expr in settings.get("confounders", {}).items():
        conf[name] = expression(parse_expression(expr)[0])

    base = pd.DataFrame({"facility_id": grid.get_level_values(0), "period": grid.get_level_values(1)})
    base["month_index"] = np.tile(np.arange(len(pstr)), len(fac_ids))
    base["calendar_month"] = base["period"].str[4:].astype(int)
    base["year"] = base["period"].str[:4].astype(int)
    base = base.merge(fac[["facility_id", "facility_name", "district_id", "district_name", "province_id", "province_name", "national_id", "national_name"]], on="facility_id", how="left")

    panels = {}
    for o in inputs["outcomes"]:
        if o["denominator_type"] == "invalid":
            continue
        df = base.copy()
        df["y"] = expression(o["numerator_operands"])
        df["denominator"] = expression(o["denominator_operands"]) if o["denominator_type"] == "mapped_population" else np.nan
        reason = np.full(len(df), "", dtype=object)
        reason[np.isnan(df["y"])] = "numerator_not_reported"
        if o["denominator_type"] == "mapped_population":
            den = df["denominator"].to_numpy()
            ok = reason == ""
            reason[ok & np.isnan(den)] = "denominator_not_reported"
            ok = reason == ""
            reason[ok & ((den < settings["minimum_denominator"]) | (den <= 0))] = "denominator_below_minimum"
            ok = reason == ""
            with np.errstate(divide="ignore", invalid="ignore"):
                reason[ok & (df["y"].to_numpy() / den > settings["max_ratio"])] = "ratio_above_max"
        for name, vals in conf.items():
            df[f"conf_{name}"] = vals
            reason[(reason == "") & np.isnan(vals)] = "missing_confounder"
        # Late reporting: trailing months whose share of valid facilities falls below tail_ratio x median.
        share = pd.Series(reason == "").groupby(df["month_index"]).mean()
        active = share[share > 0]
        if len(active):
            median_share = active.median()
            for mi in sorted(share.index, reverse=True):
                if share[mi] < settings["tail_reporting_ratio"] * median_share:
                    reason[(df["month_index"] == mi).to_numpy() & (reason == "")] = "incomplete_recent_reporting"
                else:
                    break
        df["reason"] = reason
        panels[o["outcome_id"]] = df

    # Climate cubes: facility x month over the full lag-extended and forecast range.
    max_lag = max([o["alt_lag_max"] for o in inputs["outcomes"]] + [0])
    clim_months = pd.period_range(start - max_lag, end + settings["forecast_horizon"], freq="M")
    cstr = [_period_str(p) for p in clim_months]
    clim = values.loc[values["kind"] == "climate"]
    cubes = {}
    for uid in inputs["climate_vars"]["climate_uid"]:
        sub = clim.loc[clim["dx"] == uid].pivot_table(index="ou", columns="period", values="value", aggfunc="mean")
        cubes[uid] = sub.reindex(index=fac_ids, columns=cstr)
    district_of = fac.set_index("facility_id")["district_id"]
    district_cubes = {uid: cube.groupby(district_of).transform("mean") for uid, cube in cubes.items()}

    return {
        "months": pstr,
        "climate_months": cstr,
        "panels": panels,
        "cubes": cubes,
        "district_cubes": district_cubes,
        "facilities": fac,
        "climate_vars": inputs["climate_vars"],
        "outcomes": {o["outcome_id"]: o for o in inputs["outcomes"] if o["outcome_id"] in panels},
        "end_month": pstr[-1],
    }


def completeness_tables(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long facility-month reporting status, and a facility x outcome summary."""
    rows = []
    for oid, df in data["panels"].items():
        rows.append(df[["facility_id", "facility_name", "district_name", "period"]].assign(
            outcome_id=oid, indicator_name=data["outcomes"][oid]["indicator_name"],
            reported=df["reason"].eq("").astype(int), status=df["reason"].replace("", "valid")))
    long = pd.concat(rows, ignore_index=True)
    summary = long.groupby(["outcome_id", "indicator_name", "facility_id", "facility_name", "district_name"], as_index=False).agg(
        months_in_period=("reported", "size"), months_valid=("reported", "sum"))
    summary["pct_valid"] = 100 * summary["months_valid"] / summary["months_in_period"]
    return long, summary


def missingness_bias_table(data: dict) -> pd.DataFrame:
    """Does climate differ between reported and unreported months? (informative missingness check)."""
    out = []
    for oid, df in data["panels"].items():
        valid = df["reason"].eq("")
        # Only months within each facility's active reporting span count as 'missing'.
        idx = df.loc[valid].groupby("facility_id")["month_index"].agg(["min", "max"])
        span = df["facility_id"].map(idx["min"]).le(df["month_index"]) & df["facility_id"].map(idx["max"]).ge(df["month_index"])
        missing = span & df["reason"].isin(["numerator_not_reported", "denominator_not_reported"])
        for uid, cube in data["cubes"].items():
            x = cube.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([df["facility_id"], df["period"]])).to_numpy()
            a, b = x[valid.to_numpy() & np.isfinite(x)], x[missing.to_numpy() & np.isfinite(x)]
            if len(a) < 10 or len(b) < 10:
                smd, p = np.nan, np.nan
            else:
                sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                smd = (b.mean() - a.mean()) / sd if sd > 0 else 0.0
                p = stats.ttest_ind(a, b, equal_var=False).pvalue
            out.append({"outcome_id": oid, "climate_uid": uid, "n_reported": len(a), "n_missing": len(b),
                        "mean_exposure_reported": a.mean() if len(a) else np.nan, "mean_exposure_missing": b.mean() if len(b) else np.nan,
                        "std_mean_difference": smd, "p_value": p, "informative_missingness_flag": bool(np.isfinite(smd) and abs(smd) > 0.2)})
    return pd.DataFrame(out)


def exposure_matrix(cube: pd.DataFrame, climate_months: list[str], facility_ids: np.ndarray, periods: np.ndarray, lag_max: int) -> np.ndarray:
    """n x (L+1) lagged exposures for each (facility, period) row, using the full climate cube."""
    col_index = {p: i for i, p in enumerate(climate_months)}
    arr = cube.to_numpy(float)
    row_index = {f: i for i, f in enumerate(cube.index)}
    ri = np.array([row_index[f] for f in facility_ids])
    ci = np.array([col_index[p] for p in periods])
    out = np.full((len(ri), lag_max + 1), np.nan)
    for lag in range(lag_max + 1):
        c = ci - lag
        ok = c >= 0
        out[ok, lag] = arr[ri[ok], c[ok]]
    return out
