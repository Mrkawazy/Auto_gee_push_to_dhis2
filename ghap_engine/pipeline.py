"""Orchestration: model every outcome x climate pair, run QA, forecasts, and save a versioned run."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import ENGINE_VERSION
from .bases import cumulative_gradient, reference_crossbasis
from .forecast import climate_scenarios, future_rows, score, simulate, summarise_forecast, thresholds
from .model import (BLOCKING_FLAGS, attribution, aggregate_attribution, build_spec, dersimonian_laird, draw_cb, exposure_response_table,
                    find_mmt, fit_pair_model, lag_response_table, log_rr_ci, morans_i, reference_value, residual_autocorrelation,
                    residual_seasonality_p, row_table)
from .panel import exposure_matrix

# ---------------------------------------------------------------------------------
# Model registry and run log
# ---------------------------------------------------------------------------------


class Run:
    """A versioned model run: immutable folder ghap_outputs/runs/<run_id>, registry row and JSON-lines log."""

    def __init__(self, settings: dict):
        cfg = {k: v for k, v in settings.items() if k not in ("run_started", "climate_forecast")}
        self.settings_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
        stamp = pd.Timestamp(settings["run_started"]).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{self.settings_hash}"
        self.root = Path(settings["output_folder"])
        self.dir = self.root / "runs" / self.run_id
        longest = len(str((self.dir / "facility_month" / ("x" * 31 + ".parquet")).resolve()))
        if longest >= 260:
            raise OSError(f"Output paths would reach {longest} characters (Windows limit 260). Set settings['output_folder'] to a shorter absolute path, e.g. C:/ghap_outputs.")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.settings = settings
        (self.dir / "settings.json").write_text(json.dumps(cfg, indent=1, default=str))
        self.log("run_started", engine_version=ENGINE_VERSION, settings_hash=self.settings_hash)

    def log(self, event: str, **details):
        entry = {"time": pd.Timestamp.now().isoformat(timespec="seconds"), "run_id": self.run_id, "event": event, **details}
        self.events.append(entry)
        with open(self.root / "run_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def log_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


def save_tables(run: Run, tables: dict[str, pd.DataFrame], csv_max_rows: int = 500_000) -> pd.DataFrame:
    """Write every table as parquet (and CSV when small enough) in the run folder; refresh ghap_outputs/current."""
    index = []
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str)
        df.to_parquet(run.dir / f"{name}.parquet", index=False)
        if len(df) <= csv_max_rows:
            df.to_csv(run.dir / f"{name}.csv", index=False)
        index.append({"table": name, "rows": len(df), "columns": df.shape[1], "csv": len(df) <= csv_max_rows})
    current = run.root / "current"
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(run.dir, current)
    (current / "RUN_ID.txt").write_text(run.run_id)
    return pd.DataFrame(index)


def register_run(run: Run, inputs: dict, summary: pd.DataFrame, end_month: str, status: str = "completed") -> pd.DataFrame:
    row = {"run_id": run.run_id, "started": run.settings["run_started"], "finished": pd.Timestamp.now().isoformat(timespec="seconds"),
           "engine_version": ENGINE_VERSION, "settings_hash": run.settings_hash, "input_snapshot": inputs["snapshot_key"],
           "data_hash": inputs["snapshot_hash"], "data_retrieved": inputs["retrieved"], "analysis_end_month": end_month,
           "pairs": len(summary), "pairs_fitted": int(summary["status"].eq("fitted").sum()), "pairs_qa_pass": int(summary["qa_pass"].sum()),
           "status": status, "published": False, "reviewed_by": ""}
    path = run.root / "model_registry.csv"
    reg = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()
    reg = pd.concat([reg.loc[reg.get("run_id", pd.Series(dtype=str)) != run.run_id] if len(reg) else reg, pd.DataFrame([row])], ignore_index=True)
    reg.to_csv(path, index=False)
    return reg


# ---------------------------------------------------------------------------------
# Attribution for every outcome x climate pair
# ---------------------------------------------------------------------------------


def _total_attribution(rows, lagged, spec, fit, ref, draws) -> dict:
    y = rows["y"].to_numpy(float)
    diff = spec.cb(lagged) - reference_crossbasis(ref, spec.exposure, spec.cmat)
    an = (1 - np.exp(-(diff @ fit.beta_cb))) * y
    tot_s = (1 - np.exp(-(diff.astype(np.float32) @ draws.T.astype(np.float32)))).T @ y.astype(np.float32)
    tot, obs = an.sum(), y.sum()
    lo, hi = np.quantile(tot_s, [0.025, 0.975])
    return {"an": tot, "an_low": lo, "an_high": hi, "paf": tot / obs if obs else np.nan,
            "paf_low": lo / obs if obs else np.nan, "paf_high": hi / obs if obs else np.nan}


def _rr_at(spec, fit, pooled, ref, pct):
    x = np.percentile(pooled, [pct])
    est, se = log_rr_ci(cumulative_gradient(x, ref, spec.exposure, spec.cmat), fit.beta_cb, fit.cov_cb)
    return float(np.exp(est[0])), float(np.exp(est[0] - 1.96 * se[0])), float(np.exp(est[0] + 1.96 * se[0])), float(est[0])


def _quick_effects(rows, lagged, settings, lag_max, is_rate, confs, rng, n_draws=300) -> dict:
    spec, fit, _ = fit_pair_model(rows, lagged, settings, lag_max, is_rate, confs)
    pooled = lagged[:, 0]
    mmt, mmt_pct = find_mmt(spec, fit, pooled, settings)
    out = {"rows": len(rows), "mmt": mmt, "mmt_percentile": mmt_pct}
    for p in (10, 90):
        rr, lo, hi, _ = _rr_at(spec, fit, pooled, mmt, p)
        out.update({f"rr_p{p}": rr, f"rr_p{p}_low": lo, f"rr_p{p}_high": hi})
    out.update(_total_attribution(rows, lagged, spec, fit, mmt, draw_cb(fit, n_draws, rng)))
    return out


def analyse_pair(o: dict, valid: pd.DataFrame, uid: str, cname: str, data: dict, settings: dict, rng) -> dict:
    L = o["lag_max"]
    is_rate = o["denominator_type"] == "mapped_population"
    confs = [c for c in valid.columns if c.startswith("conf_")]
    lag_all = exposure_matrix(data["cubes"][uid], data["climate_months"], valid["facility_id"].to_numpy(), valid["period"].to_numpy(), L)
    ok = np.isfinite(lag_all).all(axis=1)
    rows, lagged = valid.loc[ok].reset_index(drop=True), lag_all[ok]
    counts = {"outcome_id": o["outcome_id"], "climate_uid": uid, "excluded_missing_exposure_history": int((~ok).sum()), "rows_with_exposure_history": int(ok.sum())}
    summary = {"outcome_id": o["outcome_id"], "indicator_name": o["indicator_name"], "climate_uid": uid, "climate_name": cname,
               "denominator_type": o["denominator_type"], "is_negative_control": o["is_negative_control"], "lag_max": L,
               "rows_analysis": len(rows), "facilities": rows["facility_id"].nunique(), "months": rows["period"].nunique(),
               "first_period": rows["period"].min() if len(rows) else "", "last_period": rows["period"].max() if len(rows) else "",
               "events": float(rows["y"].sum()), "status": "fitted", "qa_flags": "", "qa_pass": False, "error": ""}
    result = {"counts": counts, "summary": summary, "ok": ok}
    if len(rows) == 0 or summary["events"] < settings["min_events"] or summary["months"] < settings["min_months"]:
        summary["status"] = "insufficient_data"
        return result
    pooled = lagged[:, 0]
    if len(np.unique(pooled)) <= 3:
        summary["status"] = "degenerate_exposure"
        return result
    flags = []
    try:
        spec, fit, use = fit_pair_model(rows, lagged, settings, L, is_rate, confs)
    except np.linalg.LinAlgError:
        summary.update(status="fit_failed", qa_flags="rank_deficient")
        return result
    except Exception as exc:  # keep the run going; the failure is reported in the summary and run log
        summary.update(status="fit_failed", qa_flags="fit_failed", error=f"{type(exc).__name__}: {exc}"[:300])
        return result
    if not (np.all(np.isfinite(fit.beta_cb)) and np.all(np.isfinite(fit.cov_cb))):
        summary.update(status="fit_failed", qa_flags="non_finite_estimates")
        return result

    mmt, mmt_pct = find_mmt(spec, fit, pooled, settings)
    cuts = tuple(np.percentile(pooled, settings["extreme_percentiles"]))
    refs = {"mmt": mmt, **{k: reference_value(k, pooled, mmt) for k in settings["reference_sensitivity"]}}
    draws = draw_cb(fit, settings["draws"], rng)
    mu = np.zeros(len(rows))
    mu[use] = fit.mu
    att = attribution(rows, lagged, spec, fit, mmt, cuts, draws)
    tag = {"outcome_id": o["outcome_id"], "climate_uid": uid}
    result["facility_month"] = row_table(rows, lagged, att, mu, mmt).assign(**tag, climate_name=cname)
    result["aggregates"] = aggregate_attribution(rows, att, mu, is_rate).assign(**tag)
    result["exposure_response"] = exposure_response_table(spec, fit, pooled, refs).assign(**tag)
    result["lag_response"] = lag_response_table(spec, fit, pooled, mmt).assign(**tag, reference_value=mmt)
    result["reference_sensitivity"] = pd.DataFrame(
        [{"reference_type": k, "reference_value": r, **_total_attribution(rows, lagged, spec, fit, r, draws)} for k, r in refs.items()]).assign(**tag)

    # Effects, test and diagnostics
    rr = {}
    for p in (1, 10, 90, 99):
        v = _rr_at(spec, fit, pooled, mmt, p)
        rr.update({f"rr_p{p}": v[0], f"rr_p{p}_low": v[1], f"rr_p{p}_high": v[2]})
        if p in (10, 90):
            if abs(v[3]) > settings["max_abs_log_rr"]:
                flags.append("implausible_effect")
            if v[1] > 0 and v[2] / v[1] > settings["max_ci_ratio"]:
                flags.append("uninformative_ci")
    wald = float(fit.beta_cb @ np.linalg.pinv(fit.cov_cb) @ fit.beta_cb)
    used = rows.loc[use]
    acf1, acf12 = residual_autocorrelation(used, fit.pearson)
    seas_p = residual_seasonality_p(used, fit.pearson)
    moran, moran_p = morans_i(data["facilities"], used, fit.pearson, rng)
    tot = result["aggregates"].query("level == 'national' and period_type == 'total'").iloc[0]
    if summary["months"] < settings["short_series_months"]:
        flags.append("short_series")
    if np.isfinite(acf1) and abs(acf1) > 0.3:
        flags.append("residual_autocorrelation")
    if np.isfinite(seas_p) and seas_p < 0.01:
        flags.append("residual_seasonality")
    if np.isfinite(moran_p) and moran_p < 0.05:
        flags.append("residual_spatial_clustering")
    if o.get("mapping_warning"):  # blocks publication until a reviewer accepts the mapping
        flags.append("mapping_warning_accepted" if o["outcome_id"] in settings.get("accepted_mapping_warnings", []) else "mapping_warning")
    summary.update({
        "rows_used_in_fit": int(use.sum()), "facilities_without_events": int(rows.loc[~use, "facility_id"].nunique()),
        "family": fit.family, "dispersion": fit.dispersion, "nb_alpha": fit.nb_alpha, "covariance": fit.cov_type,
        "trend_df": spec.trend.df, "exposure_df": spec.exposure.df, "event_terms": ";".join(spec.events), "parameters": fit.n_params,
        "qaic": fit.qaic, "pseudo_r2": fit.pseudo_r2, "mmt": mmt, "mmt_percentile": mmt_pct,
        "extreme_low_cut": cuts[0], "extreme_high_cut": cuts[1], **rr,
        "wald_chi2": wald, "wald_df": fit.p_cb, "wald_p": float(stats.chi2.sf(wald, fit.p_cb)),
        "total_an": tot["an"], "total_an_low": tot["an_low"], "total_an_high": tot["an_high"],
        "paf": tot["paf"], "paf_low": tot["paf_low"], "paf_high": tot["paf_high"],
        "residual_acf_lag1": acf1, "residual_acf_lag12": acf12, "residual_seasonality_p": seas_p, "morans_i": moran, "morans_i_p": moran_p,
    })
    summary["qa_flags"] = ";".join(flags)
    summary["qa_pass"] = not (set(flags) & BLOCKING_FLAGS)
    result["model"] = {"spec": spec, "fit": fit, "ref": mmt, "lag_max": L}

    # Heterogeneity between districts (two-stage: district fits, DerSimonian-Laird random effects, BLUPs)
    het_rows = []
    local = dict(settings, cluster_min_groups=10**9)
    for did, sub_idx in rows.groupby("district_id").groups.items():
        sub = rows.loc[sub_idx]
        if sub["y"].sum() < settings["min_events"] or sub["period"].nunique() < settings["min_months"]:
            continue
        try:
            sd = build_spec(sub, settings, None, L, is_rate, confs)
            sd.exposure, sd.cmat = spec.exposure, spec.cmat
            sd, fd, _ = fit_pair_model(sub, lagged[sub_idx], local, L, is_rate, confs, spec=sd, family_override=(fit.family, fit.nb_alpha))
        except Exception:
            continue
        rec = {"district_id": did, "district_name": sub["district_name"].iloc[0], "facilities": sub["facility_id"].nunique(), "events": sub["y"].sum()}
        for p in (10, 90):
            g = cumulative_gradient(np.percentile(pooled, [p]), mmt, spec.exposure, spec.cmat)
            est, se = log_rr_ci(g, fd.beta_cb, fd.cov_robust[-fd.p_cb:, -fd.p_cb:])
            rec[f"log_rr_p{p}"], rec[f"se_p{p}"] = float(est[0]), float(se[0])
        if np.isfinite(rec["se_p90"]) and rec["se_p90"] > 0:
            het_rows.append(rec)
    if len(het_rows) >= 2:
        h = pd.DataFrame(het_rows)
        for p in (10, 90):
            if not (h[f"se_p{p}"] > 0).all():  # e.g. P10 equals the reference for zero-inflated exposures
                continue
            m = dersimonian_laird(h[f"log_rr_p{p}"].to_numpy(), h[f"se_p{p}"].to_numpy() ** 2)
            h[f"rr_p{p}"] = np.exp(h[f"log_rr_p{p}"])
            h[f"rr_p{p}_low"], h[f"rr_p{p}_high"] = np.exp(h[f"log_rr_p{p}"] - 1.96 * h[f"se_p{p}"]), np.exp(h[f"log_rr_p{p}"] + 1.96 * h[f"se_p{p}"])
            h[f"blup_rr_p{p}"] = np.exp(m["blup"])
            h[f"blup_rr_p{p}_low"], h[f"blup_rr_p{p}_high"] = np.exp(m["blup"] - 1.96 * m["blup_se"]), np.exp(m["blup"] + 1.96 * m["blup_se"])
            h[f"pooled_rr_p{p}"] = np.exp(m["pooled"])
            h[f"i2_p{p}"], h[f"q_p{p}"], h[f"q_pvalue_p{p}"], h[f"tau2_p{p}"] = m["i2"], m["q"], m["q_p"], m["tau2"]
        result["heterogeneity"] = h.assign(**tag)
        summary["heterogeneity_i2_p90"] = float(h["i2_p90"].iloc[0])
        summary["heterogeneity_q_pvalue_p90"] = float(h["q_pvalue_p90"].iloc[0])

    # Sensitivity refits: alternative lag window and district-mean exposure (exposure measurement error)
    if settings["run_sensitivity_refits"]:
        sens = [{"specification": "main", "lag_max": L, "exposure_scale": "facility", **{k: summary[k] for k in ["rr_p10", "rr_p10_low", "rr_p10_high", "rr_p90", "rr_p90_low", "rr_p90_high", "paf", "paf_low", "paf_high", "mmt"]}, "an": summary["total_an"], "rows": len(rows)}]
        for label, cube, lag in [("alternative_lag_window", data["cubes"][uid], o["alt_lag_max"]), ("district_mean_exposure", data["district_cubes"][uid], L)]:
            la = exposure_matrix(cube, data["climate_months"], valid["facility_id"].to_numpy(), valid["period"].to_numpy(), lag)
            ok2 = np.isfinite(la).all(axis=1)
            r2 = valid.loc[ok2].reset_index(drop=True)
            try:
                if r2["y"].sum() < settings["min_events"] or r2["period"].nunique() < settings["min_months"]:
                    raise ValueError("insufficient data")
                eff = _quick_effects(r2, la[ok2], settings, lag, is_rate, confs, rng)
                sens.append({"specification": label, "lag_max": lag, "exposure_scale": "district_mean" if "district" in label else "facility", **eff})
            except Exception as exc:
                sens.append({"specification": label, "lag_max": lag, "error": str(exc)[:200]})
        result["sensitivity"] = pd.DataFrame(sens).assign(**tag)
    return result


TABLE_KEYS = ["aggregates", "exposure_response", "lag_response", "reference_sensitivity", "heterogeneity", "sensitivity"]


def _model_outcome(oi: int, oid: str, data: dict, settings: dict, fm_dir: str) -> dict:
    """Model all climate variables for one outcome (runs in a worker process when n_jobs > 1)."""
    o = data["outcomes"][oid]
    valid = data["panels"][oid].loc[lambda d: d["reason"].eq("")].reset_index(drop=True)
    cv = data["climate_vars"]
    out = {"summary": [], "counts": [], "models": {}, "masks": {}, **{k: [] for k in TABLE_KEYS}}
    fm = []
    for ci, (uid, cname) in enumerate(zip(cv["climate_uid"], cv["climate_name"])):
        rng = np.random.default_rng([settings["seed"], oi, ci])  # reproducible regardless of order or parallelism
        res = analyse_pair(o, valid, uid, cname, data, settings, rng)
        out["summary"].append(res["summary"])
        out["counts"].append(res["counts"])
        out["masks"][(oid, uid)] = res["ok"]
        for k in TABLE_KEYS:
            if k in res:
                out[k].append(res[k])
        if "facility_month" in res:
            fm.append(res["facility_month"])
        if "model" in res:
            out["models"][(oid, uid)] = res["model"]
    if fm:
        pd.concat(fm, ignore_index=True).to_parquet(Path(fm_dir) / f"{oid}.parquet", index=False)
    return out


def run_attribution(data: dict, settings: dict, run: Run, missingness: pd.DataFrame | None = None) -> dict:
    collect = {k: [] for k in ["summary", "counts"] + TABLE_KEYS}
    models, masks = {}, {}
    fm_dir = run.dir / "facility_month"
    fm_dir.mkdir(exist_ok=True)
    cv = data["climate_vars"]
    t0 = time.time()
    oids = list(data["outcomes"])
    n_jobs = max(1, int(settings.get("n_jobs", 1)))
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(n_jobs, len(oids))) as pool:
            futures = [pool.submit(_model_outcome, i, oid, data, settings, str(fm_dir)) for i, oid in enumerate(oids)]
            parts = []
            for i, f in enumerate(futures):
                parts.append(f.result())
                print(f"Modelled outcome {i + 1} / {len(oids)} ({time.time() - t0:.0f}s)")
    else:
        parts = []
        for i, oid in enumerate(oids):
            parts.append(_model_outcome(i, oid, data, settings, str(fm_dir)))
            print(f"Modelled {data['outcomes'][oid]['indicator_name'][:50]}: {len(cv)} climate variables ({time.time() - t0:.0f}s)")
    for part in parts:
        for k in collect:
            collect[k].extend(part[k])
        models.update(part["models"])
        masks.update(part["masks"])
    summary = pd.DataFrame(collect["summary"])
    for _, r in summary.loc[summary["error"].ne("")].iterrows():
        run.log("pair_fit_failed", outcome_id=r["outcome_id"], climate_uid=r["climate_uid"], error=r["error"])

    # Negative control: a climate association with the utilisation tracer suggests access-to-care effects.
    nc = summary.loc[summary["is_negative_control"] & summary["status"].eq("fitted")]
    sig = nc.loc[(nc["rr_p90_low"] > 1) | (nc["rr_p90_high"] < 1) | (nc["rr_p10_low"] > 1) | (nc["rr_p10_high"] < 1), "climate_uid"]
    if missingness is not None and len(missingness):
        im = missingness.loc[missingness["informative_missingness_flag"], ["outcome_id", "climate_uid"]].assign(_im=True)
        summary = summary.merge(im, on=["outcome_id", "climate_uid"], how="left")
    else:
        summary["_im"] = False
    def add(flags, new):
        return ";".join([f for f in [flags, new] if f])
    mask_nc = summary["climate_uid"].isin(set(sig)) & ~summary["is_negative_control"] & summary["status"].eq("fitted")
    summary.loc[mask_nc, "qa_flags"] = summary.loc[mask_nc, "qa_flags"].apply(lambda f: add(f, "access_effect_in_negative_control"))
    mask_im = summary["_im"].eq(True) & summary["status"].eq("fitted")
    summary.loc[mask_im, "qa_flags"] = summary.loc[mask_im, "qa_flags"].apply(lambda f: add(f, "informative_missingness"))
    summary = summary.drop(columns="_im")
    for (oid, uid), m in models.items():
        m["qa_pass"] = bool(summary.loc[(summary["outcome_id"] == oid) & (summary["climate_uid"] == uid), "qa_pass"].iloc[0])

    negative_control = nc[["outcome_id", "indicator_name", "climate_uid", "climate_name", "rr_p10", "rr_p10_low", "rr_p10_high", "rr_p90", "rr_p90_low", "rr_p90_high", "wald_p"]].copy()
    negative_control["access_effect_detected"] = negative_control["climate_uid"].isin(set(sig))
    tables = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in collect.items() if k not in ("summary", "counts")}
    names = cv.set_index("climate_uid")["climate_name"]
    for k, df in tables.items():
        if len(df) and "climate_uid" in df.columns:
            df.insert(2, "climate_name", df["climate_uid"].map(names))
            df.insert(1, "indicator_name", df["outcome_id"].map({k2: v["indicator_name"] for k2, v in data["outcomes"].items()}))
    tables["model_summary"] = summary
    tables["negative_control"] = negative_control
    run.log("attribution_completed", pairs=len(summary), fitted=int(summary["status"].eq("fitted").sum()), qa_pass=int(summary["qa_pass"].sum()))
    return {"tables": tables, "counts": pd.DataFrame(collect["counts"]), "models": models, "masks": masks, "facility_month_dir": fm_dir}


def load_facility_month(results: dict, outcome_id: str | None = None) -> pd.DataFrame:
    files = sorted(Path(results["facility_month_dir"]).glob("*.parquet"))
    if outcome_id:
        files = [f for f in files if f.stem == outcome_id]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# ---------------------------------------------------------------------------------
# Forecasts and rolling-origin evaluation
# ---------------------------------------------------------------------------------

FAC_COLS = ["facility_id", "facility_name", "district_id", "district_name", "province_id", "province_name", "national_id", "national_name"]


def _external(settings, uid):
    ext = settings.get("climate_forecast")
    if ext is None or not len(ext):
        return None
    sub = ext.loc[ext["climate_variable"] == uid]
    return sub if len(sub) else None


def _forecast_pair(fut, model, data, uid, origin_period, horizon, mask, n_draws, rng, settings):
    scen = climate_scenarios(data["cubes"][uid], data["climate_months"], origin_period, horizon, mask, _external(settings, uid))
    spec = model["spec"]
    cbs = [spec.cb(exposure_matrix(sc, data["climate_months"], fut["facility_id"].to_numpy(), fut["period"].to_numpy(), model["lag_max"])) for sc in scen]
    cb_ref = reference_crossbasis(model["ref"], spec.exposure, spec.cmat)
    return simulate(fut, spec, model["fit"], cbs, n_draws, rng, cb_ref)


def _forecast_outcome(oi: int, oid: str, data: dict, settings: dict, pairs: dict, masks: dict) -> dict:
    """Forecasts, thresholds and rolling-origin scores for one outcome (worker-safe)."""
    rng = np.random.default_rng([settings["seed"], 10_000 + oi])
    out = {"pair_fc": [], "outcome_fc": None, "thr": None, "cv_rows": [], "log": []}
    o = data["outcomes"][oid]
    H = settings["forecast_horizon"]
    months = data["months"]
    T = len(months) - 1
    origin_period = months[-1]
    fac = data["facilities"][FAC_COLS]
    is_rate = o["denominator_type"] == "mapped_population"
    valid = data["panels"][oid].loc[lambda d: d["reason"].eq("")].reset_index(drop=True)
    confs = [c for c in valid.columns if c.startswith("conf_")]
    if valid["y"].sum() < settings["min_events"] or valid["period"].nunique() < settings["min_months"]:
        out["log"].append(("forecast_skipped", {"outcome_id": oid, "reason": "insufficient data"}))
        return out
    thr = thresholds(valid, settings)
    out["thr"] = thr.assign(outcome_id=oid, indicator_name=o["indicator_name"])
    fut = future_rows(valid, fac, origin_period, T, H, confs, is_rate)
    try:
        bspec, bfit, _ = fit_pair_model(valid, None, settings, 0, is_rate, confs)
    except Exception as exc:
        out["log"].append(("baseline_fit_failed", {"outcome_id": oid, "error": str(exc)[:300]}))
        return out
    y, mu, an = simulate(fut, bspec, bfit, [np.empty((len(fut), 0))], settings["forecast_draws"], rng, None)
    baseline_fc = summarise_forecast(fut, y, mu, an, thr, is_rate)
    fcs = {}
    for uid, m in pairs.items():
        yp, mp, ap = _forecast_pair(fut, m, data, uid, origin_period, H, False, settings["forecast_draws"], rng, settings)
        fcs[uid] = summarise_forecast(fut, yp, mp, ap, thr, is_rate)
        out["pair_fc"].append(fcs[uid].assign(outcome_id=oid, climate_uid=uid))

    # Rolling-origin evaluation: forecasts issued at past origins, scored against what was then observed.
    cv_rows = out["cv_rows"]
    obs_map = valid.set_index(["facility_id", "period"])["y"]
    for i in range(settings["cv_origins"]):
        O = T - H - 3 * i
        train = valid.loc[valid["month_index"] <= O]
        if O < 0 or train["period"].nunique() < settings["min_months"] or train["y"].sum() < settings["min_events"]:
            continue
        op = months[O]
        fcv = future_rows(train, fac, op, O, H, confs, is_rate)
        obs = obs_map.reindex(pd.MultiIndex.from_arrays([fcv["facility_id"], fcv["period"]])).to_numpy()
        has = np.isfinite(obs)
        if not has.any():
            continue
        base_rows = {"outcome_id": oid, "origin": op, "lead_months": fcv["lead_months"].to_numpy()[has]}

        def record(model_name, ydraw):
            cv_rows.append(pd.DataFrame({**base_rows, "model": model_name, **score(ydraw[has], obs[has])}))

        try:
            bs, bf, _ = fit_pair_model(train, None, settings, 0, is_rate, confs, family_override=(bfit.family, bfit.nb_alpha))
            record("no_climate_baseline", simulate(fcv, bs, bf, [np.empty((len(fcv), 0))], settings["cv_draws"], rng, None)[0])
        except Exception:
            continue
        naive = obs_map.reindex(pd.MultiIndex.from_arrays([fcv["facility_id"], [(pd.Period(f"{p[:4]}-{p[4:]}", freq="M") - 12).strftime("%Y%m") for p in fcv["period"]]])).to_numpy()
        nv = has & np.isfinite(naive)
        if nv.any():
            cv_rows.append(pd.DataFrame({"outcome_id": oid, "origin": op, "lead_months": fcv["lead_months"].to_numpy()[nv], "model": "seasonal_naive",
                                         "crps": np.abs(naive[nv] - obs[nv]), "abs_error": np.abs(naive[nv] - obs[nv]), "error": naive[nv] - obs[nv],
                                         "cover80": np.nan, "cover95": np.nan}))
        for uid, m in pairs.items():
            rows_p = valid.loc[masks[(oid, uid)]]
            tr = rows_p.loc[rows_p["month_index"] <= O].reset_index(drop=True)
            if tr["period"].nunique() < settings["min_months"] or tr["y"].sum() < settings["min_events"]:
                continue
            lag_tr = exposure_matrix(data["cubes"][uid], data["climate_months"], tr["facility_id"].to_numpy(), tr["period"].to_numpy(), m["lag_max"])
            try:
                sp, fp, _ = fit_pair_model(tr, lag_tr, settings, m["lag_max"], is_rate, confs, family_override=(m["fit"].family, m["fit"].nb_alpha))
            except Exception:
                continue
            mm = {"spec": sp, "fit": fp, "ref": m["ref"], "lag_max": m["lag_max"]}
            record(f"{uid}|operational", _forecast_pair(fcv, mm, data, uid, op, H, True, settings["cv_draws"], rng, settings)[0])
            record(f"{uid}|perfect_climate", _forecast_pair(fcv, mm, data, uid, op, H, False, settings["cv_draws"], rng, dict(settings, climate_forecast=None))[0])

    sel = select_forecast_model(cv_rows, settings)
    chosen = fcs[sel["selected_climate_uid"]] if sel["selected_climate_uid"] else baseline_fc
    out["outcome_fc"] = chosen.assign(outcome_id=oid, indicator_name=o["indicator_name"], **sel)
    return out


def run_forecasts(data: dict, settings: dict, results: dict, run: Run) -> dict:
    origin_period = data["months"][-1]
    pair_fc, outcome_fc, thr_all, cv_rows = [], [], [], []
    t0 = time.time()
    oids = list(data["outcomes"])
    pairs_by = {oid: {uid: m for (o2, uid), m in results["models"].items() if o2 == oid and m["qa_pass"]} for oid in oids}
    masks_by = {oid: {k: v for k, v in results["masks"].items() if k[0] == oid} for oid in oids}
    n_jobs = max(1, int(settings.get("n_jobs", 1)))
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(n_jobs, len(oids))) as pool:
            futures = [pool.submit(_forecast_outcome, i, oid, data, settings, pairs_by[oid], masks_by[oid]) for i, oid in enumerate(oids)]
            parts = [f.result() for f in futures]
    else:
        parts = []
        for i, oid in enumerate(oids):
            parts.append(_forecast_outcome(i, oid, data, settings, pairs_by[oid], masks_by[oid]))
            print(f"Forecasts done for {data['outcomes'][oid]['indicator_name'][:50]} ({time.time() - t0:.0f}s)")
    for part in parts:
        pair_fc += part["pair_fc"]
        cv_rows += part["cv_rows"]
        if part["thr"] is not None:
            thr_all.append(part["thr"])
        if part["outcome_fc"] is not None:
            outcome_fc.append(part["outcome_fc"])
        for event, details in part["log"]:
            run.log(event, **details)

    names = data["climate_vars"].set_index("climate_uid")["climate_name"]
    tag = {"issue_month": origin_period, "run_id": run.run_id}
    forecast_pair = pd.concat(pair_fc, ignore_index=True).assign(**tag) if pair_fc else pd.DataFrame()
    if len(forecast_pair):
        forecast_pair.insert(0, "climate_name", forecast_pair["climate_uid"].map(names))
        forecast_pair.insert(0, "indicator_name", forecast_pair["outcome_id"].map({k: v["indicator_name"] for k, v in data["outcomes"].items()}))
    forecast_outcome = pd.concat(outcome_fc, ignore_index=True).assign(**tag) if outcome_fc else pd.DataFrame()
    if len(forecast_outcome):
        forecast_outcome["selected_climate_name"] = forecast_outcome["selected_climate_uid"].map(names).fillna("")
    cv = pd.concat(cv_rows, ignore_index=True) if cv_rows else pd.DataFrame(columns=["outcome_id", "model", "lead_months", "crps", "abs_error", "error", "cover80", "cover95", "origin"])
    skill = forecast_skill_table(cv, data, names)
    run.log("forecasts_completed", outcomes=len(outcome_fc), pair_forecasts=len(pair_fc))
    return {"forecast_outcome": forecast_outcome, "forecast_pair": forecast_pair, "forecast_thresholds": pd.concat(thr_all, ignore_index=True) if thr_all else pd.DataFrame(),
            "forecast_skill": skill, "forecast_cv_scores": cv}


def select_forecast_model(outcome_cv: list[pd.DataFrame], settings: dict) -> dict:
    """Pick a climate model only if it beats the no-climate model clearly and consistently out of sample.

    With many climate variables one will beat the baseline by chance, so a candidate needs
    (a) skill >= `min_forecast_skill` and (b) a one-sided paired t-test on row-level CRPS differences
    with p below 0.05 / number of candidates (Bonferroni).
    """
    out = {"selected_model": "no_climate_baseline", "selected_climate_uid": "", "cv_crps_selected": np.nan, "cv_crps_baseline": np.nan,
           "skill_vs_baseline": 0.0, "selection_p_value": np.nan}
    if not outcome_cv:
        return out
    cvo = pd.concat(outcome_cv, ignore_index=True)
    base = cvo.loc[cvo["model"] == "no_climate_baseline"]
    if base.empty:
        return out
    out["cv_crps_selected"] = out["cv_crps_baseline"] = base_crps = float(base["crps"].mean())
    candidates = [m for m in cvo["model"].unique() if m.endswith("|operational")]
    best = None
    for m in candidates:
        sub = cvo.loc[cvo["model"] == m]
        if len(sub) != len(base):
            continue  # rows align by origin and facility only when both models were scored everywhere
        diff = base["crps"].to_numpy() - sub["crps"].to_numpy()
        skill = 1 - sub["crps"].mean() / base_crps if base_crps > 0 else 0.0
        p = float(stats.ttest_1samp(diff, 0.0, alternative="greater").pvalue) if np.std(diff) > 0 else 1.0
        if skill >= settings["min_forecast_skill"] and p < 0.05 / max(1, len(candidates)) and (best is None or skill > best[1]):
            best = (m, skill, p, float(sub["crps"].mean()))
    if best:
        out.update(selected_model="climate_model", selected_climate_uid=best[0].split("|")[0], cv_crps_selected=best[3], skill_vs_baseline=best[1], selection_p_value=best[2])
    return out


def forecast_skill_table(cv: pd.DataFrame, data: dict, names: pd.Series) -> pd.DataFrame:
    if cv.empty:
        return pd.DataFrame()
    cv = cv.copy()
    cv["lead_band"] = pd.cut(cv["lead_months"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"]).astype(str)
    out = []
    for band, sub in [("all", cv)] + list(cv.groupby("lead_band")):
        g = sub.groupby(["outcome_id", "model"]).agg(origins=("origin", "nunique"), scored_rows=("crps", "size"), crps=("crps", "mean"),
                                                     mae=("abs_error", "mean"), bias=("error", "mean"), coverage_80=("cover80", "mean"), coverage_95=("cover95", "mean")).reset_index()
        base = g.loc[g["model"] == "no_climate_baseline"].set_index("outcome_id")["crps"]
        g["skill_vs_baseline"] = 1 - g["crps"] / g["outcome_id"].map(base)
        out.append(g.assign(lead_band=band))
    sk = pd.concat(out, ignore_index=True)
    sk["climate_uid"] = sk["model"].str.split("|").str[0].where(sk["model"].str.contains(r"\|"), "")
    sk["forecast_type"] = sk["model"].str.split("|").str[1].where(sk["model"].str.contains(r"\|"), sk["model"])
    sk["climate_name"] = sk["climate_uid"].map(names).fillna("")
    sk.insert(1, "indicator_name", sk["outcome_id"].map({k: v["indicator_name"] for k, v in data["outcomes"].items()}))
    return sk
