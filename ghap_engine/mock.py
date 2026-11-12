"""Synthetic DHIS2 server for testing the full notebook without network access (GHAP_MOCK_DHIS2=1).

The synthetic truth: malaria positivity rises with temperature at lags 1-3 and with rainfall;
preterm births rise with third-trimester heat; live births (negative control) have no climate effect.
Zeros are not stored (as in DHIS2) but the form is marked submitted; some forms are missing and the
latest month is partially reported.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


def uid(seed: str) -> str:
    h = hashlib.sha1(seed.encode()).hexdigest()
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    n = int(h, 16)
    chars = [alphabet[n % 52]]
    n //= 52
    for _ in range(10):
        chars.append(alphabet[n % 62])
        n //= 62
    return "".join(chars)


TRUE_MALARIA_LOG_RR_PER_C = 0.06  # per +1 C of temperature, cumulative over lags 1-3
TRUE_PRETERM_LOG_RR_PER_C = 0.04  # per +1 C, cumulative over lags 0-2

CLIMATE = {"temp": "Air temperature (ERA5-Land)", "rain": "Precipitation (ERA5-Land)", "hw": "Heatwave count (ERA5-Land)", "rh": "Relative humidity (ERA5-Land)"}
DE = {k: uid(f"de_{k}") for k in ["mal_pos_a", "mal_pos_b", "mal_test_a", "mal_test_b", "preterm", "births_a", "births_b", "still", "mdeath", "cholera"]}
CLIM_UID = {k: uid(f"clim_{k}") for k in CLIMATE}
DATASET_CLIMATE = "qqND8igwWqz"
DATASET_HMIS = uid("ds_hmis")


class MockDHIS2Client:
    base = "https://mock.dhis2.local"

    def __init__(self, seed: int = 7, n_facilities: int = 24, start: str = "2018-01", end: str = "2026-10"):
        rng = np.random.default_rng(seed)
        self.posts: list[tuple[str, dict, dict]] = []
        self.stored: dict = {}
        self.cocs: dict = {}
        self.metadata: dict = {}
        self.completions: dict = {}
        prov = [("P1", uid("prov1"), "Mashonaland Central"), ("P2", uid("prov2"), "Mashonaland East")]
        dists = [(uid(f"dist{i}"), f"District {i}", prov[i % 2]) for i in range(4)]
        self.national = uid("national")
        self.facilities = []
        for i in range(n_facilities):
            d = dists[i % 4]
            self.facilities.append({"id": uid(f"fac{i}"), "name": f"mc Facility {i:02d}", "level": 4,
                                    "geometry": {"type": "Point", "coordinates": [31.0 + rng.normal(0, 0.4), -17.0 + rng.normal(0, 0.4)]},
                                    "ancestors": [{"id": self.national, "name": "Zimbabwe", "level": 1}, {"id": d[2][1], "name": d[2][2], "level": 2},
                                                  {"id": d[0], "name": d[1], "level": 3}]})
        self.facilities.append({"id": uid("polyfac"), "name": "Polygon facility", "level": 4, "geometry": {"type": "Polygon", "coordinates": [[[31, -17], [31.1, -17], [31, -17.1]]]}, "ancestors": []})
        months = pd.period_range(start, end, freq="M")
        self.periods = [p.strftime("%Y%m") for p in months]
        nm = len(months)
        cal = np.array([p.month for p in months])
        year_anom = rng.normal(0, 1.2, size=nm // 12 + 2)
        values = []
        fac_ids = [f["id"] for f in self.facilities[:-1]]
        temp = {}
        for j, f in enumerate(fac_ids):
            elev = rng.normal(0, 1.5)
            seas = 21 + 4.5 * np.cos(2 * np.pi * (cal - 1) / 12)
            anom = np.zeros(nm)
            for t in range(nm):
                anom[t] = (0.5 * anom[t - 1] if t else 0) + rng.normal(0, 0.9)
            t_series = seas + elev + anom + np.repeat(year_anom, 12)[:nm] * 0.6
            temp[f] = t_series
            rain = np.clip(rng.gamma(2, 1, nm) * (60 + 70 * np.cos(2 * np.pi * (cal - 1) / 12)), 0, None)
            hw = rng.poisson(np.clip((t_series - 23), 0, None) * 0.6)
            rh = 60 + 15 * np.cos(2 * np.pi * (cal - 1) / 12) + rng.normal(0, 4, nm)
            for k, series in {"temp": t_series, "rain": rain, "hw": hw, "rh": rh}.items():
                for t in range(nm):
                    values.append((CLIM_UID[k], self.periods[t], f, float(np.round(series[t], 3))))
        self.climate_values = values
        health = []
        reports = []
        self.health_start = 12  # health data from 2019-01
        for j, f in enumerate(fac_ids):
            size = rng.lognormal(4.0, 0.5)
            ts = temp[f]
            mean_t = ts.mean()
            for t in range(self.health_start, nm):
                if rng.random() < 0.06:
                    continue  # form not submitted: genuinely missing
                if t == nm - 1 and rng.random() < 0.6:
                    continue  # most facilities have not yet reported the latest month
                season = 0.5 * np.cos(2 * np.pi * (cal[t] - 2) / 12)
                tests = rng.poisson(size * 3 * (1 + 0.3 * np.cos(2 * np.pi * (cal[t] - 2) / 12)))
                lagavg = np.mean([ts[t - l] - mean_t for l in (1, 2, 3)])
                pos_rate = np.clip(0.12 * np.exp(season + TRUE_MALARIA_LOG_RR_PER_C * lagavg * 3 / 3 * 1.0), 0, 0.95)
                pos = rng.negative_binomial(20, 20 / (20 + tests * pos_rate)) if tests else 0
                pos = min(pos, tests)
                births = rng.poisson(size * 0.8)
                tri = np.mean([ts[t - l] - mean_t for l in (0, 1, 2)])
                pre = rng.poisson(births * 0.09 * np.exp(TRUE_PRETERM_LOG_RR_PER_C * tri)) if births else 0
                pre = min(pre, births)
                still = rng.binomial(births, 0.015) if births else 0
                md = rng.poisson(0.03)
                chol = rng.poisson(0.4 * np.exp(0.002 * (np.mean([0]))))
                split = rng.random()
                vals = {"mal_pos_a": int(pos * split), "mal_pos_b": pos - int(pos * split), "mal_test_a": int(tests * 0.7), "mal_test_b": tests - int(tests * 0.7),
                        "preterm": pre, "births_a": int(births * 0.6), "births_b": births - int(births * 0.6), "still": still, "mdeath": md, "cholera": chol}
                for k, v in vals.items():
                    if v != 0:  # DHIS2 does not store zeros
                        health.append((DE[k], self.periods[t], f, float(v)))
                reports.append((f"{DATASET_HMIS}.ACTUAL_REPORTS", self.periods[t], f, 1.0))
        self.health_values = health
        self.report_values = reports
        self.store = pd.DataFrame(values + health + reports, columns=["dx", "pe", "ou", "value"])
        self.store_index = self.store.set_index(["dx"]).sort_index()

    # -- API ----------------------------------------------------------------
    def get(self, endpoint: str, params=None):
        params = params or {}
        if endpoint.startswith("me"):
            return {"id": "mockuser"}
        if endpoint.startswith('sections/'):
            return self.metadata['sections'][endpoint.split('/')[1]]
        if (endpoint == 'dataElements' and self.metadata.get('dataElements')
                and str(params.get('filter', '')).startswith('id:in:')):
            ids = str(params['filter']).split('[', 1)[-1].rstrip(']').split(',')
            return {'dataElements': [v for key, v in self.metadata.get('dataElements', {}).items() if key in ids]}
        if endpoint in {'categoryOptions', 'categories', 'categoryCombos', 'dataSets', 'sections'} or (endpoint == 'dataElements' and str(params.get('filter', '')).startswith('code:')):
            codes = str(params.get('filter', '')).split('[', 1)[-1].rstrip(']').split(',')
            return {endpoint: [v for v in self.metadata.get(endpoint, {}).values() if v.get('code') in codes]}
        if endpoint == 'organisationUnits':
            ids = str(params.get('filter', '')).split('[', 1)[-1].rstrip(']').split(',')
            return {'organisationUnits': [{'id': value, 'openingDate': '1980-01-01'} for value in ids]}
        if endpoint == 'completeDataSetRegistrations':
            units = params.get('orgUnit', [])
            return {'completeDataSetRegistrations': [v for (ds, ou, pe), v in self.completions.items()
                    if ds == params['dataSet'] and pe == params['period'] and ou in units]}
        if endpoint == 'scheduling/running/types':
            return []
        if endpoint.startswith('system/tasks/ANALYTICS_TABLE/'):
            return [{'level': 'INFO', 'completed': True, 'message': 'Analytics tables updated: success'}]
        if endpoint == 'dataValueSets':
            elements = {e['dataElement']['id'] for e in self.metadata['dataSets'][params['dataSet']]['dataSetElements']}
            return {'dataValues': [{'dataElement': de, 'period': pe, 'orgUnit': ou, 'categoryOptionCombo': co, 'value': value}
                    for (de, pe, ou, co), value in self.stored.items() if ou == params['orgUnit'] and de in elements]}
        if endpoint.startswith("dataSets/"):
            if endpoint.split('/')[1] in self.metadata.get('dataSets', {}):
                return self.metadata['dataSets'][endpoint.split('/')[1]]
            return {"id": DATASET_CLIMATE, "name": "Climate",
                    "dataSetElements": [{"dataElement": {"id": CLIM_UID[k], "name": n, "shortName": n[:50], "aggregationType": "AVERAGE"}} for k, n in CLIMATE.items()],
                    "organisationUnits": self.facilities}
        if endpoint == "dataElements":
            ids = re.search(r"\[(.*)\]", params["filter"]).group(1).split(",")
            names = {v: k for k, v in DE.items()}
            return {"dataElements": [{"id": i, "name": f"HMIS {names[i]}", "dataSetElements": [{"dataSet": {"id": DATASET_HMIS}}]} for i in ids if i in names]}
        if endpoint == "analytics.json":
            if isinstance(params, dict):
                dims = {v.split(':', 1)[0]: v.split(':', 1)[1] for v in params['dimension']}
                parts = dims['dx'].split('.')
                stored = self.stored.get((parts[0], dims['pe'], dims['ou'], parts[1] if len(parts) > 1 else None))
                return {'headers': [{'name': key} for key in ['dx', 'ou', 'pe', 'value']],
                        'rows': [[dims['dx'], dims['ou'], dims['pe'], stored]] if stored is not None else []}
            dims = {p[1].split(":")[0]: p[1].split(":")[1].split(";") for p in params if p[0] == "dimension"}
            sub = self.store.loc[self.store["dx"].isin(dims["dx"]) & self.store["pe"].isin(dims["pe"]) & self.store["ou"].isin(dims["ou"])]
            return {"headers": [{"name": "dx"}, {"name": "pe"}, {"name": "ou"}, {"name": "value"}], "rows": sub.astype(str).values.tolist()}
        if endpoint.startswith("categoryCombos/"):
            cc = endpoint.split("/")[1]
            return {"id": cc, "categoryOptionCombos": self.cocs.get(cc, [])}
        if endpoint == "dataValues":
            key = (params["de"], params["pe"], params["ou"], params.get("co"))
            return [self.stored[key]] if key in self.stored else []
        raise KeyError(endpoint)

    def post(self, endpoint: str, json: dict, params=None):
        params = params or {}
        self.posts.append((endpoint, json, params))
        if endpoint == "metadata":
            if params.get('importMode') == 'VALIDATE':
                return {'status': 'OK', 'stats': {'created': 0}}
            for kind, objects in json.items():
                self.metadata.setdefault(kind, {}).update({item['id']: item for item in objects})
            for cc in json.get("categoryCombos", []):
                opts = [o["id"] for c in json["categories"] if c["id"] in [x["id"] for x in cc["categories"]] for o in c["categoryOptions"]]
                self.cocs[cc["id"]] = [{"id": uid(f"coc_{o}"), "categoryOptions": [{"id": o}]} for o in opts]
            return {"status": "OK", "response": {"stats": {"created": sum(len(v) for v in json.values() if isinstance(v, list)), "updated": 0, "ignored": 0}}}
        if endpoint == 'completeDataSetRegistrations':
            for item in json['completeDataSetRegistrations']:
                self.completions[(item['dataSet'], item['organisationUnit'], item['period'])] = item
            return {'status': 'SUCCESS', 'importCount': {'imported': len(json['completeDataSetRegistrations'])}}
        if endpoint == 'resourceTables/analytics':
            return {'status': 'OK', 'response': {'id': 'mockAnalyticsTask'}}
        if endpoint.startswith('maintenance/categoryOptionComboUpdate/categoryCombo/'):
            combo_id = endpoint.rsplit('/', 1)[-1]
            combo = self.metadata['categoryCombos'][combo_id]
            options = [option['id'] for category in combo['categories']
                       for option in self.metadata['categories'][category['id']]['categoryOptions']]
            self.cocs[combo_id] = [{'id': uid('coc_' + option), 'categoryOptions': [{'id': option}]} for option in options]
            return {'status': 'OK'}
        if endpoint == "dataValueSets":
            if str(params.get("dryRun", "false")) != "true":
                for dv in json["dataValues"]:
                    self.stored[(dv["dataElement"], dv["period"], dv["orgUnit"], dv.get("categoryOptionCombo"))] = dv["value"]
            return {"status": "SUCCESS", "importCount": {"imported": len(json["dataValues"]), "updated": 0, "ignored": 0, "deleted": 0}, "conflicts": []}
        return {"status": "OK"}

    def post_once(self, endpoint, json, params=None):
        return self.post(endpoint, json, params)

    def delete(self, endpoint, params=None):
        if endpoint.startswith('sections/'):
            section_id = endpoint.split('/')[1]
            self.metadata['sections'].pop(section_id, None)
            for dataset in self.metadata.get('dataSets', {}).values():
                dataset['sections'] = [s for s in dataset.get('sections', []) if s['id'] != section_id]
            return {}
        if endpoint == 'completeDataSetRegistrations':
            self.completions.pop((params['ds'], params['ou'], params['pe']), None)
            return {}
        raise KeyError(endpoint)


def write_mock_workbook(path: str | Path) -> Path:
    path = Path(path)
    df = pd.DataFrame({
        "Indicator name": ["Malaria positive cases (VHW & HFs)", "Preterm births", "HIS - D3 - Total Still births at health facility",
                           "HIS - D3 - Grand Total Maternal Deaths", "HIS - D3 - Total Live births at Hospital/clinic", "RDNS - Cholera Cases"],
        "Numerator": [f"{DE['mal_pos_a']} + {DE['mal_pos_b']}", DE["preterm"], DE["still"], DE["mdeath"], f"{DE['births_a']} + {DE['births_b']}", DE["cholera"]],
        "Denominator": [f"{DE['mal_test_a']} + {DE['mal_test_b']}", f"{DE['births_a']} + {DE['births_b']}", f"{DE['births_a']} + {DE['births_b']}", 1, 1, 1],
        "Denominator description": ["Malaria tests done", "Total births", "Total births including still births", "", "", ""],
    })
    df.to_excel(path, index=False)
    return path
