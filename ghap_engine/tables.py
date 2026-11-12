"""Service-layer tables: descriptive views, row accounting and the data dictionary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import LEVELS
from .panel import REASONS


def _scaled_level_series(df: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    """Org-unit monthly totals from valid facility rows, scaled to full reporting (months with >=50% reporting)."""
    v = df.loc[df["reason"].eq("")]
    total_fac = df.groupby(id_col)["facility_id"].nunique()
    g = v.groupby([id_col, name_col, "period", "year", "calendar_month"], as_index=False).agg(
        observed=("y", "sum"), denominator=("denominator", "sum"), reporting_facilities=("facility_id", "nunique"))
    g["expected_facilities"] = g[id_col].map(total_fac)
    g["reporting_share"] = g["reporting_facilities"] / g["expected_facilities"]
    g["observed_scaled"] = np.where(g["reporting_share"] >= 0.5, g["observed"] / g["reporting_share"], np.nan)
    return g.rename(columns={id_col: "org_unit_id", name_col: "org_unit_name"})


def outcome_trends(data: dict) -> pd.DataFrame:
    """Historical monthly series per org unit level: observed, scaled observed, denominator and rate."""
    frames = []
    for oid, df in data["panels"].items():
        rate = data["outcomes"][oid]["denominator_type"] == "mapped_population"
        for level, id_col, name_col in LEVELS:
            g = _scaled_level_series(df, id_col, name_col)
            g["rate_per_1000"] = 1000 * g["observed"] / g["denominator"] if rate else np.nan
            if not rate:
                g["denominator"] = np.nan
            frames.append(g.assign(level=level, outcome_id=oid, indicator_name=data["outcomes"][oid]["indicator_name"]))
    cols = ["outcome_id", "indicator_name", "level", "org_unit_id", "org_unit_name", "period", "year", "calendar_month", "observed",
            "observed_scaled", "denominator", "rate_per_1000", "reporting_facilities", "expected_facilities", "reporting_share"]
    return pd.concat(frames, ignore_index=True)[cols]


def outcome_seasonality(trends: pd.DataFrame) -> pd.DataFrame:
    """Monthly climatology of outcomes: mean, P10 and P90 by calendar month across years."""
    g = trends.dropna(subset=["observed_scaled"]).groupby(["outcome_id", "indicator_name", "level", "org_unit_id", "org_unit_name", "calendar_month"])
    out = g.agg(years=("year", "nunique"), mean_cases=("observed_scaled", "mean"),
                p10_cases=("observed_scaled", lambda s: s.quantile(0.1)), p90_cases=("observed_scaled", lambda s: s.quantile(0.9)),
                mean_rate_per_1000=("rate_per_1000", "mean")).reset_index()
    return out


def _climate_long(data: dict) -> pd.DataFrame:
    fac = data["facilities"].set_index("facility_id")
    analysis = set(data["months"])
    frames = []
    for uid, cube in data["cubes"].items():
        s = cube.stack(future_stack=True).rename("value").reset_index()
        s.columns = ["facility_id", "period", "value"]
        s = s.loc[s["period"].isin(analysis)].dropna(subset=["value"])
        frames.append(s.assign(climate_uid=uid))
    long = pd.concat(frames, ignore_index=True)
    for c in ["facility_name", "district_id", "district_name", "province_id", "province_name", "national_id", "national_name"]:
        long[c] = long["facility_id"].map(fac[c])
    long["year"] = long["period"].str[:4].astype(int)
    long["calendar_month"] = long["period"].str[4:].astype(int)
    return long


def climate_tables(data: dict, settings: dict) -> dict:
    """Climate trends, climatology, hazard thresholds and facility-month hazard flags."""
    long = _climate_long(data)
    names = data["climate_vars"].set_index("climate_uid")["climate_name"]
    long["climate_name"] = long["climate_uid"].map(names)
    lo_p, hi_p = settings["extreme_percentiles"]
    thr = long.groupby(["climate_uid", "climate_name"])["value"].quantile([lo_p / 100, hi_p / 100]).unstack()
    thr.columns = ["low_hazard_threshold", "high_hazard_threshold"]
    thr = thr.reset_index()
    clim = long.groupby(["climate_uid", "facility_id", "calendar_month"])["value"].agg(["mean", "std"])
    key = pd.MultiIndex.from_arrays([long["climate_uid"], long["facility_id"], long["calendar_month"]])
    long["climatology"] = clim["mean"].reindex(key).to_numpy()
    long["anomaly"] = long["value"] - long["climatology"]
    with np.errstate(divide="ignore", invalid="ignore"):
        long["anomaly_z"] = long["anomaly"] / clim["std"].reindex(key).to_numpy()
    t = thr.set_index("climate_uid")
    long["high_hazard"] = (long["value"] > long["climate_uid"].map(t["high_hazard_threshold"])).astype(int)
    long["low_hazard"] = (long["value"] < long["climate_uid"].map(t["low_hazard_threshold"])).astype(int)

    trends, season = [], []
    for level, id_col, name_col in LEVELS:
        g = long.groupby(["climate_uid", "climate_name", id_col, name_col, "period", "year", "calendar_month"], as_index=False).agg(
            mean_value=("value", "mean"), min_value=("value", "min"), max_value=("value", "max"), mean_anomaly=("anomaly", "mean"),
            facilities=("facility_id", "nunique"), facilities_high_hazard=("high_hazard", "sum"), facilities_low_hazard=("low_hazard", "sum"))
        g = g.rename(columns={id_col: "org_unit_id", name_col: "org_unit_name"}).assign(level=level)
        g["share_high_hazard"] = g["facilities_high_hazard"] / g["facilities"]
        g["share_low_hazard"] = g["facilities_low_hazard"] / g["facilities"]
        trends.append(g)
        s = g.groupby(["climate_uid", "climate_name", "level", "org_unit_id", "org_unit_name", "calendar_month"]).agg(
            years=("year", "nunique"), mean_value=("mean_value", "mean"), p10_value=("mean_value", lambda v: v.quantile(0.1)),
            p90_value=("mean_value", lambda v: v.quantile(0.9))).reset_index()
        season.append(s)
    hazard_events = long[["climate_uid", "climate_name", "facility_id", "facility_name", "district_name", "period", "value", "climatology",
                          "anomaly", "anomaly_z", "high_hazard", "low_hazard"]]
    return {"climate_trends": pd.concat(trends, ignore_index=True), "climate_seasonality": pd.concat(season, ignore_index=True),
            "hazard_thresholds": thr, "hazard_events": hazard_events}


def funnel_table(data: dict, pair_counts: pd.DataFrame, model_summary: pd.DataFrame) -> pd.DataFrame:
    """Row accounting from the full facility x month grid to rows eligible for publication."""
    rows = []
    for oid, df in data["panels"].items():
        counts = df["reason"].value_counts()
        base = {"outcome_id": oid, "indicator_name": data["outcomes"][oid]["indicator_name"], "grid_facility_months": len(df)}
        for r in list(REASONS)[:6]:
            base[f"excluded_{r}"] = int(counts.get(r, 0))
        base["valid_outcome_rows"] = int((df["reason"] == "").sum())
        rows.append(base)
    out = pd.DataFrame(rows)
    pc = pair_counts.merge(model_summary[["outcome_id", "climate_uid", "status", "qa_pass", "rows_used_in_fit", "facilities_without_events"]],
                           on=["outcome_id", "climate_uid"], how="left")
    return out.merge(pc, on="outcome_id", how="right")


COLUMN_DESCRIPTIONS = {
    "outcome_id": "Stable outcome identifier (indicator slug + hash of its formula)",
    "indicator_name": "Indicator name from the mapping workbook",
    "climate_uid": "DHIS2 UID of the climate data element",
    "climate_name": "Climate data element name",
    "level": "Org unit level of the row: facility, district, province or national",
    "org_unit_id": "DHIS2 org unit UID",
    "org_unit_name": "Org unit name",
    "period": "Month (YYYYMM), year (YYYY) or 'all' for the whole analysis period",
    "period_type": "month, year or total",
    "observed": "Observed count (sum of facility values)",
    "observed_scaled": "Observed count scaled to full facility reporting (months with at least 50% reporting)",
    "denominator": "Population at risk (mapped denominator); empty for count outcomes",
    "expected": "Model-fitted expected count",
    "expected_without_exposure": "Expected count if the exposure history had been at the reference value",
    "rr": "Cumulative relative risk of the month's exposure history versus the reference (lags 0..L)",
    "rr_low": "Lower 95% limit (Monte Carlo)",
    "rr_high": "Upper 95% limit (Monte Carlo)",
    "af": "Attributable fraction = 1 - 1/RR (negative = fewer cases than at the reference)",
    "an": "Attributable number of cases = AF x observed (SUM across units and months is valid)",
    "paf": "Population attributable fraction = sum(AN) / sum(observed) within the group (never average)",
    "attributable_rate_per_1000": "Attributable cases per 1,000 population at risk (rate outcomes only)",
    "an_above_ref": "Attributable cases from exposures above the reference (e.g. heat)",
    "an_below_ref": "Attributable cases from exposures below the reference (e.g. cold)",
    "an_extreme_high": "Attributable cases from exposures above max(P90, reference) (high hazard)",
    "an_extreme_low": "Attributable cases from exposures below min(P10, reference) (low hazard)",
    "reference_exposure": "Reference exposure (minimum-risk exposure unless stated otherwise)",
    "reference_type": "mmt (minimum-risk exposure), median, p25 or p75",
    "exposure_lag0": "Climate value in the outcome month",
    "exposure_window_mean": "Mean climate value across the lag window",
    "exposure_percentile": "Percentile of the pooled exposure distribution",
    "exposure_value": "Exposure value at that percentile",
    "lag": "Months between exposure and outcome",
    "facility_months": "Number of facility-months in the group",
    "forecast_mean": "Mean of the predictive distribution of cases",
    "forecast_median": "Median forecast cases",
    "forecast_pi80_low": "80% prediction interval lower bound",
    "forecast_pi80_high": "80% prediction interval upper bound",
    "forecast_pi95_low": "95% prediction interval lower bound",
    "forecast_pi95_high": "95% prediction interval upper bound",
    "expected_cases": "Mean expected cases (no outcome noise)",
    "climate_attributable_cases": "Forecast cases attributable to forecast climate versus the reference",
    "climate_attributable_fraction": "climate_attributable_cases / expected_cases",
    "forecast_denominator": "Projected denominator (facility same-month mean of the last 2 years)",
    "forecast_rate_per_1000": "forecast_mean per 1,000 projected denominator",
    "alert_threshold": "Alert threshold: calendar-month 75th percentile of history",
    "epidemic_threshold": "Epidemic threshold: calendar-month mean + 2 SD of history",
    "prob_exceed_alert": "Probability the month's cases exceed the alert threshold",
    "prob_exceed_epidemic": "Probability the month's cases exceed the epidemic threshold",
    "risk_level": "high (P(epidemic) >= 0.5), moderate (P(alert) >= 0.5), watch (P(alert) >= 0.25), low",
    "lead_months": "Months ahead of the forecast origin",
    "issue_month": "Last month of data used for the forecast",
    "run_id": "Model run identifier (model registry)",
    "status": "fitted, insufficient_data, degenerate_exposure, fit_failed",
    "qa_pass": "True when the fit has no blocking QA flag",
    "qa_flags": "Semicolon-separated QA flags",
    "crps": "Continuous ranked probability score (lower is better)",
    "skill_vs_baseline": "1 - CRPS(model) / CRPS(no-climate model); positive = climate improves forecasts",
}


TABLE_DESCRIPTIONS = {
    "mapping_validation": "Workbook indicators, parsed DHIS2 operands and names, lag windows, negative-control flag and mapping warnings",
    "completeness_facility_month": "Reporting status of every facility-month per outcome (valid or exclusion reason)",
    "completeness_summary": "Share of valid months per facility and outcome",
    "missingness_bias": "Climate in reported vs unreported months (informative missingness check)",
    "outcome_trends": "Historical monthly outcome series by facility/district/province/national",
    "outcome_seasonality": "Monthly climatology of outcomes (mean, P10, P90 by calendar month)",
    "climate_trends": "Monthly climate series by level with hazard counts",
    "climate_seasonality": "Monthly climatology of each climate variable by level",
    "hazard_thresholds": "Pooled P10/P90 hazard thresholds per climate variable",
    "hazard_events": "Facility-month climate values, anomalies and hazard flags",
    "model_summary": "One row per outcome x climate pair: fit, family, MMT, RRs, PAF, diagnostics, QA flags",
    "exposure_response": "Cumulative exposure-response curve (RR by exposure percentile) vs MMT and alternative references",
    "lag_response": "Lag-specific RR at P1/P10/P90/P99 vs MMT",
    "aggregates": "Attributable numbers, PAF and attributable rates by level and month/year/total with valid draw-based CIs",
    "attribution_facility_month": "Facility-month RR, AF, AN (+ components) with 95% CIs (one parquet file per outcome in facility_month/)",
    "reference_sensitivity": "Whole-period AN and PAF under alternative reference exposures",
    "heterogeneity": "District-specific and BLUP RRs with I2 and Cochran Q",
    "sensitivity": "Refits with an alternative lag window and district-mean exposure",
    "negative_control": "Climate associations with the utilisation tracer (live births)",
    "forecast_outcome": "Next 12 months: selected model forecast per outcome at every level, with intervals, thresholds and risk",
    "forecast_pair": "Next 12 months: forecast from every QA-passed outcome x climate model",
    "forecast_thresholds": "Alert and epidemic thresholds per org unit and calendar month",
    "forecast_skill": "Rolling-origin forecast skill (CRPS, MAE, coverage) vs no-climate and seasonal-naive baselines",
    "forecast_cv_scores": "Row-level rolling-origin scores",
    "funnel": "Row accounting from the facility x month grid to modelled and QA-passed pairs",
    "run_log": "Events logged during this run",
    "data_dictionary": "Column descriptions for every table",
}


def data_dictionary(tables: dict[str, pd.DataFrame], descriptions: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            base = col
            for suffix in ("_low", "_high"):
                if col.endswith(suffix) and col not in COLUMN_DESCRIPTIONS:
                    base = col[: -len(suffix)]
            desc = COLUMN_DESCRIPTIONS.get(col) or (COLUMN_DESCRIPTIONS.get(base, "") + (" (95% interval bound)" if base != col else ""))
            rows.append({"table": name, "table_description": descriptions.get(name, ""), "column": col, "dtype": str(df[col].dtype), "description": desc.strip()})
    return pd.DataFrame(rows)
