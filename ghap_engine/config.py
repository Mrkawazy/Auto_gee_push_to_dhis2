"""Default settings and outcome-specific modelling rules."""

from __future__ import annotations

import pandas as pd

# Keyword rules (matched case-insensitively against the indicator name) that set the
# lag window per outcome. The first matching rule wins; `default` applies otherwise.
# `lag_max` is the main window; `alt_lag_max` is refitted as a sensitivity analysis.
LAG_RULES = [
    # Rainfall/temperature act on malaria through vector breeding with 1-4 month delays.
    {"keywords": ["malaria"], "lag_max": 4, "alt_lag_max": 6, "rationale": "vector breeding and incubation delay (0-4 months)"},
    # Cholera responds to rainfall/flooding within weeks; a short window is used.
    {"keywords": ["cholera"], "lag_max": 2, "alt_lag_max": 4, "rationale": "waterborne transmission responds within weeks"},
    # Birth outcomes: third trimester (framework section 5) with full pregnancy as sensitivity.
    {"keywords": ["preterm", "still", "birth", "lowbirth", "low birth", "newborn", "neonat", "maternal", "obstetric"],
     "lag_max": 2, "alt_lag_max": 8, "rationale": "third trimester (lags 0-2); full pregnancy (0-8) as sensitivity"},
]
DEFAULT_LAG_RULE = {"lag_max": 2, "alt_lag_max": 4, "rationale": "default short window"}


def default_settings() -> dict:
    """Return the full settings dictionary used by every pipeline step."""
    today = pd.Timestamp.now()
    return {
        # ---- data sources ------------------------------------------------------
        "workbook": "DHIS2 Health Outcomes.xlsx",  # mapping of indicators to DHIS2 numerator/denominator operands
        "workbook_sheet": None,  # None = first sheet containing the expected columns
        "workbook_columns": {},  # optional explicit {'name':..., 'numerator':..., 'denominator':...}
        "input_dataset": "qqND8igwWqz",  # climate dataset; its assigned point facilities define the study units
        "start_month": "2015-01",  # earliest month requested; the panel is trimmed to months with data
        "end_month": None,  # None = last complete month minus `reporting_lag_months`
        "reporting_lag_months": 1,  # late-reporting buffer: Nov run -> analysis ends in Sep
        "tail_reporting_ratio": 0.8,  # trailing months with reporting < 80% of the median are dropped
        "district_level": 3,  # DHIS2 org unit level of districts
        "province_level": 2,  # DHIS2 org unit level of provinces
        "zero_fill_when_form_submitted": True,  # blank component in a submitted form = 0 (DHIS2 does not store zeros)
        "minimum_denominator": 1,  # rows with a smaller mapped denominator are excluded
        "max_ratio": 1.0,  # rows with numerator/denominator above this are excluded as data errors
        "confounders": {},  # {'name': 'uid + uid'} DHIS2 expressions retrieved per facility-month
        "event_periods": {"covid19_disruption": ("2020-04", "2021-12")},  # indicator dummies for known disruptions
        "negative_control_keywords": ["live birth"],  # utilisation tracer used to detect access-to-care effects
        # ---- model -----------------------------------------------------------------
        "exposure_df": 3,  # natural cubic spline df for exposure-response (framework: 3 to 5)
        "fourier_harmonics": 2,  # seasonal harmonics (replaces 4 df/year calendar spline)
        "trend_years_per_df": 3,  # one long-term trend df per 3 years of data (minimum 1 = linear)
        "trend_events_per_df": 100,  # and at most one trend df per 100 events (prevents overfitted trends for rare outcomes)
        "min_events": 30,  # minimum total events per outcome/climate pair
        "min_months": 24,  # minimum distinct months per pair
        "short_series_months": 36,  # below this the pair is flagged `short_series`
        "cluster_min_groups": 20,  # cluster-robust (facility) covariance needs at least this many facilities
        "overdispersion_threshold": 1.2,  # Pearson dispersion above this triggers quasi-Poisson / NB choice
        "mmt_search_percentiles": (5, 95),  # minimum-risk reference searched inside this pooled exposure range
        "extreme_percentiles": (10, 90),  # hazard components: extreme low (<P10) and extreme high (>P90)
        "reference_sensitivity": ["median", "p25", "p75"],  # alternative references (framework section 10)
        "run_sensitivity_refits": True,  # refit with alternative lag window and district-mean exposure
        "draws": 1000,  # Monte Carlo draws for attribution uncertainty
        "seed": 42,
        "n_jobs": 1,  # >1 models outcomes in parallel worker processes (same results; e.g. 4 on an 8-core laptop)
        "max_abs_log_rr": 3.0,  # |log RR| at P90/P10 above this (RR > 20) fails QA as implausible
        "max_ci_ratio": 100.0,  # RR upper/lower CI ratio above this fails QA as uninformative
        # ---- forecasting ----------------------------------------------------------
        "forecast_horizon": 12,
        "forecast_draws": 1000,
        "cv_origins": 4,  # rolling-origin evaluation origins (3 months apart)
        "cv_draws": 200,
        "min_forecast_skill": 0.02,  # a climate model must cut out-of-sample CRPS by >= 2% vs the no-climate model
        "climate_forecast": None,  # optional DataFrame of seasonal forecasts (facility_id, period, climate_variable, member, value)
        "alert_percentile": 75,  # alert threshold: facility-calendar-month P75
        "epidemic_sd": 2.0,  # epidemic threshold: mean + 2 SD
        # ---- outputs ---------------------------------------------------------------
        "output_folder": "ghap_outputs",  # runs saved under ghap_outputs/runs/<run_id>, latest copy in ghap_outputs/current
        "publish_start_month": "2024-01",  # earliest month of attribution values sent to DHIS2
        # attribution parameters stored per facility-month (all parameters exist as DHIS2 elements; add more here to send them)
        "publish_facility_params": ["expected", "expected_without_exposure", "an", "an_low", "an_high", "af", "af_low", "af_high", "rr", "rr_low", "rr_high", "an_above_ref", "an_below_ref"],
        "publish_complete": True,
        "publish_analytics": True,  # aggregate analytics refresh is server-wide, not dataset-specific
        "analytics_timeout_seconds": 3600,
        "analytics_poll_seconds": 5,
        "analytics_skip_outliers": True,  # required for separate DHIS2 analytics databases
        "publish_max_values": 5_000_000,  # total-run guard; individual requests remain batched
        "accepted_mapping_warnings": [],  # outcome_ids whose mapping warning a reviewer has accepted (otherwise blocks QA)
        "run_started": today.isoformat(timespec="seconds"),
    }


def lag_rule(indicator_name: str) -> dict:
    """Return the lag rule for an indicator name."""
    lowered = indicator_name.lower()
    for rule in LAG_RULES:
        if any(k in lowered for k in rule["keywords"]):
            return rule
    return DEFAULT_LAG_RULE


def analysis_end_month(settings: dict) -> pd.Period:
    """Last month included in modelling (excludes the current, incomplete month and the reporting lag)."""
    if settings.get("end_month"):
        return pd.Period(settings["end_month"], freq="M")
    current = pd.Period(pd.Timestamp(settings["run_started"]), freq="M")
    return current - 1 - int(settings["reporting_lag_months"])
