"""DHIS2 publishing: metadata, data values, readback verification (gated by review and dry-run).

Design
- One data element per (outcome, parameter); the climate variable is a category ("GHAP climate
  variable"), so DHIS2 visualisations can filter by climate variable instead of 1,000s of elements.
- Additive counts (AN, components, expected) use aggregationType SUM and are stored at facilities only;
  DHIS2 sums them correctly to any level.
- Ratios and interval bounds (RR, AF, PAF, CIs, rates) use aggregationType NONE and are stored at every
  level from the engine's draw-based aggregation, so DHIS2 never averages ratios or CI bounds.
- Forecasts (selected model per outcome) go to a separate monthly dataset with open future periods.
- Every value carries the model run_id in its comment for provenance.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import numpy as np
import pandas as pd
from .publish_lifecycle import (save_json, check_import, prepare_metadata, verify_values,
                                reopen_forms, complete_forms, refresh_analytics)

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

PARAM_LABELS = {
    'expected': 'Historical fitted cases', 'an': 'Climate-attributable cases',
    'expected_without_exposure': 'Fitted cases at reference climate',
    'an_low': 'Attributable cases - lower 95% bound', 'an_high': 'Attributable cases - upper 95% bound',
    'af': 'Attributable fraction', 'af_low': 'Attributable fraction - lower 95% bound',
    'af_high': 'Attributable fraction - upper 95% bound', 'rr': 'Relative risk',
    'rr_low': 'Relative risk - lower 95% bound', 'rr_high': 'Relative risk - upper 95% bound',
    'an_above_ref': 'Attributable cases above reference', 'an_below_ref': 'Attributable cases below reference',
    'paf': 'Aggregate attributable fraction', 'paf_low': 'Aggregate fraction - lower 95% bound',
    'paf_high': 'Aggregate fraction - upper 95% bound', 'forecast_mean': 'Forecast cases',
    'forecast_pi80_low': 'Forecast - lower 80% prediction bound', 'forecast_pi80_high': 'Forecast - upper 80% prediction bound',
    'forecast_pi95_low': 'Forecast - lower 95% prediction bound', 'forecast_pi95_high': 'Forecast - upper 95% prediction bound',
    'prob_exceed_alert': 'Probability above alert threshold', 'prob_exceed_epidemic': 'Probability above epidemic threshold',
    'alert_threshold': 'Alert threshold', 'epidemic_threshold': 'Epidemic threshold',
    'climate_attributable_cases': 'Forecast climate-attributable cases',
    'selected_climate_name': 'Selected climate exposure (forecast model)',
}

# parameter: (short code, aggregation type, levels where stored)
ATTRIBUTION_PARAMS = {
    "an": ("AN", "SUM", "facility"),
    "an_above_ref": ("ANhi", "SUM", "facility"),
    "an_below_ref": ("ANlo", "SUM", "facility"),
    "an_extreme_high": ("ANxh", "SUM", "facility"),
    "an_extreme_low": ("ANxl", "SUM", "facility"),
    "expected": ("EXP", "SUM", "facility"),
    "expected_without_exposure": ("EXPREF", "SUM", "facility"),
    "an_low": ("ANl95", "NONE", "all"),
    "an_high": ("ANu95", "NONE", "all"),
    "af": ("AF", "NONE", "facility"),
    "af_low": ("AFl95", "NONE", "facility"),
    "af_high": ("AFu95", "NONE", "facility"),
    "rr": ("RR", "NONE", "facility"),
    "rr_low": ("RRl95", "NONE", "facility"),
    "rr_high": ("RRu95", "NONE", "facility"),
    "paf": ("PAF", "NONE", "aggregate"),
    "paf_low": ("PAFl95", "NONE", "aggregate"),
    "paf_high": ("PAFu95", "NONE", "aggregate"),
    "attributable_rate_per_1000": ("AR1k", "NONE", "aggregate"),
    "exposure_lag0": ("EXPO", "AVERAGE", "facility"),
    "reference_exposure": ("REF", "NONE", "facility"),
}
FORECAST_PARAMS = {
    "forecast_mean": ("FCm", "SUM", "facility"),
    "climate_attributable_cases": ("FCan", "SUM", "facility"),
    "forecast_pi80_low": ("FCl80", "NONE", "all"),
    "forecast_pi80_high": ("FCu80", "NONE", "all"),
    "forecast_pi95_low": ("FCl95", "NONE", "all"),
    "forecast_pi95_high": ("FCu95", "NONE", "all"),
    "prob_exceed_alert": ("FCpa", "NONE", "all"),
    "prob_exceed_epidemic": ("FCpe", "NONE", "all"),
    "alert_threshold": ("FCta", "NONE", "all"),
    "epidemic_threshold": ("FCte", "NONE", "all"),
}

FORECAST_TEXT_PARAMS = {'selected_climate_name': ('FCclim', 'NONE', 'all')}


def forecast_climate_names(frame, climate_vars):
    """Resolve selected exposure from its UID; distinguish explicit baseline models."""
    names = dict(zip(climate_vars['climate_uid'], climate_vars['climate_name']))
    labels = []
    for row in frame.to_dict('records'):
        selected = row.get('selected_climate_uid')
        if row.get('selected_model') == 'no_climate_baseline':
            if isinstance(selected, str) and selected.strip():
                raise ValueError('Baseline forecast unexpectedly has a selected climate UID.')
            labels.append('No climate exposure (baseline model)')
        elif isinstance(selected, str) and selected in names:
            label = names[selected]
            supplied = row.get('selected_climate_name')
            if isinstance(supplied, str) and supplied.strip() and supplied != label:
                raise ValueError('Forecast selected climate name does not match its UID: ' + selected)
            labels.append(label)
        else:
            raise ValueError('Forecast selected climate UID/model is missing or unknown; refusing an unlabelled forecast.')
    return pd.Series(labels, index=frame.index, dtype=str)


def make_uid(code: str) -> str:
    n = int(hashlib.sha1(code.encode()).hexdigest(), 16)
    chars = [ALPHABET[n % 52]]
    n //= 52
    for _ in range(10):
        chars.append(ALPHABET[n % 62])
        n //= 62
    return "".join(chars)


def _digest(text: str, n: int = 6) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:n].upper()


def build_metadata(data: dict, settings: dict) -> tuple[dict, dict]:
    """Metadata payload and a lookup {(table, outcome_id, param): data element uid}."""
    cv = data["climate_vars"]
    options = [{"id": make_uid(f"GHAP2_CLIM_{u}"), "code": f"GHAP2_CLIM_{u}", "name": f"GHAP {n}"[:230], "shortName": f"GHAP clim {u}"}
               for u, n in zip(cv["climate_uid"], cv["climate_name"])]
    category = {"id": make_uid("GHAP2_CAT_CLIMATE"), "code": "GHAP2_CAT_CLIMATE", "name": "GHAP climate variable", "shortName": "GHAP climate variable",
                "dataDimensionType": "DISAGGREGATION", "dataDimension": True, "categoryOptions": [{"id": o["id"]} for o in options]}
    combo = {"id": make_uid("GHAP2_CC_CLIMATE"), "code": "GHAP2_CC_CLIMATE", "name": "GHAP climate variable", "dataDimensionType": "DISAGGREGATION",
             "categories": [{"id": category["id"]}], 'skipTotal': True}
    if cv.empty or cv['climate_uid'].duplicated().any():
        raise ValueError('Climate indicators must be nonempty with unique UIDs.')
    elements, lookup, sections = [], {}, []
    selected = set(settings['publish_facility_params'])
    unknown = selected - set(ATTRIBUTION_PARAMS)
    if unknown:
        raise ValueError('Unknown publish_facility_params: ' + str(sorted(unknown)))
    att_params = {k: v for k, v in ATTRIBUTION_PARAMS.items() if k in selected or v[2] == 'aggregate'}
    dataset_ids = {'attribution': make_uid('GHAP2_DS_ATTRIBUTION'), 'forecast': make_uid('GHAP2_DS_FORECAST')}
    for oid, o in data["outcomes"].items():
        dg = _digest(oid)
        label = o["indicator_name"][:150]
        for table, params, cc in [("attribution", att_params, combo["id"]),
                                  ("forecast", {**FORECAST_PARAMS, **FORECAST_TEXT_PARAMS}, None),
                                  ('forecast_pair', FORECAST_PARAMS, combo['id'])]:
            for param, (short, agg, _) in params.items():
                suffix = '_CLIM' if table == 'forecast_pair' else ''
                code = f"GHAP2_{dg}_{short}{suffix}"
                display_param = param + (' by climate' if table == 'forecast_pair' else '')
                el = {"id": make_uid(code), "code": code, "name": f"GHAP {label} | {display_param}"[:230], "shortName": f"GHAP {dg} {short}{suffix}"[:50],
                      "formName": PARAM_LABELS.get(param, param.replace('_', ' ').capitalize()), "domainType": "AGGREGATE", "valueType": "NUMBER", "aggregationType": agg,
                      "zeroIsSignificant": True, "description": f"GHAP model output ({table}) for {o['indicator_name']}: {param}. Engine run id in value comment."}
                if cc:
                    el["categoryCombo"] = {"id": cc}
                if param in FORECAST_TEXT_PARAMS:
                    el['valueType'] = 'TEXT'
                elements.append(el)
                lookup[(table, oid, param)] = el["id"]
            code = f'GHAP2_SECTION_{table.upper()}_{dg}'
            # Keep legacy selected-model records addressable, but the visible form
            # has exactly one climate-column section per outcome, like attribution.
            if table != 'forecast':
                sections.append({'id': make_uid(code), 'code': code, 'name': o['indicator_name'][:230],
                                 'dataSet': {'id': dataset_ids['forecast' if table == 'forecast_pair' else table]}, 'sortOrder': len(sections),
                                 'dataElements': [{'id': lookup[(table, oid, param)]} for param in params]})
    fac = data["facilities"]
    org_units = set(fac["facility_id"])
    for col in ["district_id", "province_id", "national_id"]:
        org_units |= {v for v in fac[col] if not str(v).startswith("unknown")}
    ou_ref = [{"id": u} for u in sorted(org_units)]
    att_ds = {"id": make_uid("GHAP2_DS_ATTRIBUTION"), "code": "GHAP2_DS_ATTRIBUTION", "name": "GHAP Climate-Health Attribution (monthly)",
              "shortName": "GHAP attribution monthly", "periodType": "Monthly", "openFuturePeriods": 0, "expiryDays": 0,
              "dataSetElements": [{"dataElement": {"id": lookup[k]}} for k in lookup if k[0] == "attribution"], "organisationUnits": ou_ref}
    fc_ds = {"id": make_uid("GHAP2_DS_FORECAST"), "code": "GHAP2_DS_FORECAST", "name": "GHAP Climate-Health Forecast (monthly)",
             "shortName": "GHAP forecast monthly", "periodType": "Monthly", "openFuturePeriods": settings["forecast_horizon"] + 1, "expiryDays": 0,
             "dataSetElements": [{"dataElement": {"id": lookup[k]}} for k in lookup if k[0] in {"forecast", 'forecast_pair'}], "organisationUnits": ou_ref}
    for dataset in [att_ds, fc_ds]:
        dataset['sections'] = [{'id': section['id']} for section in sections if section['dataSet']['id'] == dataset['id']]
        dataset['dataElementDecoration'] = True
        dataset['description'] = ('GHAP model outputs; fractions and probabilities are decimals. Do not sum attribution across climate categories. '
                                  'Only modelled study facilities contribute to parent totals. Run provenance is recorded in value comments.')
    meta = {"categoryOptions": options, "categories": [category], "categoryCombos": [combo], "dataElements": elements,
            "dataSets": [att_ds, fc_ds], 'sections': sections}
    return meta, {"lookup": lookup, "combo": combo["id"], "options": {u: o["id"] for u, o in zip(cv["climate_uid"], options)}, 'datasets': dataset_ids}


def _fmt(v) -> str | None:
    if v is None or not np.isfinite(v):
        return None
    # DHIS2 NUMBER rejects exponent notation, even for otherwise valid small fractions.
    return format(Decimal(f"{float(v):.6g}"), 'f')


def build_data_values(results: dict, forecasts: dict, data: dict, settings: dict, ids: dict, coc_by_option: dict, run_id: str) -> pd.DataFrame:
    """Long table of DHIS2 data values for QA-passed pairs from `publish_start_month` onwards."""
    from .pipeline import load_facility_month

    summary = results["tables"]["model_summary"]
    ok_pairs = set(map(tuple, summary.loc[summary["qa_pass"], ["outcome_id", "climate_uid"]].to_numpy()))
    start = settings.get("publish_start_month", "2024-01").replace("-", "")
    frames = []
    for oid in data["outcomes"]:
        fm = load_facility_month(results, oid)
        if fm.empty:
            continue
        keep = np.array([(oid, u) in ok_pairs for u in fm["climate_uid"]], dtype=bool) & (fm["period"] >= start).to_numpy()
        fm = fm.loc[keep]
        for param, (_, _, where) in ATTRIBUTION_PARAMS.items():
            if where == "aggregate" or param not in fm.columns or param not in settings["publish_facility_params"]:
                continue
            frames.append(pd.DataFrame({"dataElement": ids["lookup"][("attribution", oid, param)], "period": fm["period"].to_numpy(),
                                        "orgUnit": fm["facility_id"].to_numpy(), "categoryOptionCombo": fm["climate_uid"].map(lambda u: coc_by_option.get(ids["options"][u])).to_numpy(),
                                        "value": fm[param].to_numpy(float)}))
    agg = results["tables"]["aggregates"]
    if len(agg):
        agg = agg.loc[agg["level"].ne("facility") & agg["period_type"].eq("month") & (agg["period"] >= start)]
        agg = agg.loc[np.array([(o, u) in ok_pairs for o, u in zip(agg["outcome_id"], agg["climate_uid"])], dtype=bool)]
        agg = agg.loc[~agg["org_unit_id"].astype(str).str.startswith("unknown")]
        for param, (_, _, where) in ATTRIBUTION_PARAMS.items():
            if where in ("all", "aggregate") and param in agg.columns and ('attribution', next(iter(data['outcomes'])), param) in ids['lookup']:
                for oid, sub in agg.groupby("outcome_id"):
                    frames.append(pd.DataFrame({"dataElement": ids["lookup"][("attribution", oid, param)], "period": sub["period"].to_numpy(),
                                                "orgUnit": sub["org_unit_id"].to_numpy(), "categoryOptionCombo": sub["climate_uid"].map(lambda u: coc_by_option.get(ids["options"][u])).to_numpy(),
                                                "value": sub[param].to_numpy(float)}))
    fc = forecasts.get("forecast_outcome", pd.DataFrame())
    if len(fc):
        blocked = {oid for oid, o in data["outcomes"].items() if o.get("mapping_warning") and oid not in settings.get("accepted_mapping_warnings", [])}
        fc = fc.loc[~fc["org_unit_id"].astype(str).str.startswith("unknown") & ~fc["outcome_id"].isin(blocked)]
        fc = fc.copy()
        fc['selected_climate_name'] = forecast_climate_names(fc, data['climate_vars'])
        for param, (_, _, where) in FORECAST_PARAMS.items():
            sub = fc if where == "all" else fc.loc[fc["level"] == "facility"]
            for oid, s2 in sub.groupby("outcome_id"):
                frames.append(pd.DataFrame({"dataElement": ids["lookup"][("forecast", oid, param)], "period": s2["period"].to_numpy(),
                                            "orgUnit": s2["org_unit_id"].to_numpy(), "categoryOptionCombo": None, "value": s2[param].to_numpy(float)}))
        for oid, s2 in fc.groupby('outcome_id'):
            frames.append(pd.DataFrame({'dataElement': ids['lookup'][('forecast', oid, 'selected_climate_name')],
                'period': s2['period'].to_numpy(), 'orgUnit': s2['org_unit_id'].to_numpy(),
                'categoryOptionCombo': None, 'value': s2['selected_climate_name'].to_numpy()}))
    pair = forecasts.get('forecast_pair', pd.DataFrame())
    if not pair.empty:
        required = {'outcome_id', 'climate_uid', 'org_unit_id', 'period', 'level'} | set(FORECAST_PARAMS)
        if required - set(pair):
            raise ValueError('Exposure forecasts are missing columns: ' + str(sorted(required - set(pair))))
        unknown = set(pair['climate_uid']) - set(ids['options'])
        if unknown or set(pair['outcome_id']) - set(data['outcomes']):
            raise ValueError('Exposure forecasts contain unknown outcomes/climates: ' + str(sorted(unknown)))
        blocked = {oid for oid, o in data['outcomes'].items() if o.get('mapping_warning')
                   and oid not in settings.get('accepted_mapping_warnings', [])}
        pair = pair.loc[[((oid, climate) in ok_pairs and oid not in blocked)
                         for oid, climate in zip(pair['outcome_id'], pair['climate_uid'])]]
        pair = pair.loc[~pair['org_unit_id'].astype(str).str.startswith('unknown')]
        for param, (_, _, where) in FORECAST_PARAMS.items():
            sub = pair if where == 'all' else pair.loc[pair['level'].eq('facility')]
            for oid, s2 in sub.groupby('outcome_id'):
                frames.append(pd.DataFrame({'dataElement': ids['lookup'][('forecast_pair', oid, param)],
                    'period': s2['period'].astype(str).to_numpy(), 'orgUnit': s2['org_unit_id'].to_numpy(),
                    'categoryOptionCombo': s2['climate_uid'].map(lambda u: coc_by_option.get(ids['options'][u])).to_numpy(),
                    'value': s2[param].to_numpy(float)}))
    if not frames:
        return pd.DataFrame(columns=["dataElement", "period", "orgUnit", "categoryOptionCombo", "value", "comment"])
    dv = pd.concat(frames, ignore_index=True)
    text_ids = {uid for (table, _, param), uid in ids['lookup'].items() if param in FORECAST_TEXT_PARAMS}
    dv["value"] = [v if de in text_ids else _fmt(v) for de, v in zip(dv['dataElement'], dv['value'])]
    dv = dv.dropna(subset=["value"])
    dv["comment"] = run_id
    return dv


def _import_summary(resp: dict) -> dict:
    r = resp.get("response", resp)
    counts = r.get("importCount") or r.get("stats") or {}
    return {"status": r.get("status", resp.get("status")), "imported": counts.get("imported", counts.get("created", 0)),
            "updated": counts.get("updated", 0), "ignored": counts.get("ignored", 0), "deleted": counts.get("deleted", 0),
            "conflicts": (r.get("conflicts") or [])[:20]}


def resolve_cocs(client, combo_id: str, expected, generate=True) -> dict:
    """Map category option uid -> category option combo uid (asks DHIS2 to generate missing combos)."""
    for attempt in range(2):
        cc = client.get(f"categoryCombos/{combo_id}", {"fields": "id,categoryOptionCombos[id,categoryOptions[id]]"})
        mapping = {c["categoryOptions"][0]["id"]: c["id"] for c in cc.get("categoryOptionCombos", []) if c.get("categoryOptions")}
        ready = len(mapping) >= expected if isinstance(expected, int) else set(expected).issubset(mapping)
        if ready:
            return mapping
        if attempt == 0 and generate:
            client.post(f"maintenance/categoryOptionComboUpdate/categoryCombo/{combo_id}", {})
    raise RuntimeError(f"Required category option combos for {combo_id} are missing.")


def publish(*args, **kwargs):
    from .publish_workflow import publish as publish_run
    from .publish_lifecycle import publication_lock
    run = kwargs.get('run') if 'run' in kwargs else args[5]
    with publication_lock(run.root):
        return publish_run(*args, **kwargs)


def run_analytics(client) -> dict:
    """Optional: start the DHIS2 analytics table update (server-wide; skip tracker/event tables)."""
    return client.post("resourceTables/analytics", {}, {"skipEvents": "true", "skipEnrollment": "true", "skipTrackedEntities": "true"})


from .publish_workflow import create_or_update_datasets
