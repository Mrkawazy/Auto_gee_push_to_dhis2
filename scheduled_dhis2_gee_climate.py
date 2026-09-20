#!/usr/bin/env python3
"""Scheduled DHIS2 -> Google Earth Engine climate extraction and upload script.

This script is intended to be run as a scheduled task from Windows Task Scheduler,
PowerShell, or a cron-based environment. It does the following:

1. Authenticates to DHIS2.
2. Downloads organisation units and keeps only valid point geometries.
3. Authenticates to Google Earth Engine.
4. Extracts ERA5-LAND climate variables for each organisation unit.
5. Resamples and prepares the climate dataset.
6. Ensures required DHIS2 data elements exist and are attached to the target dataset.
7. Uploads or updates data values to DHIS2.
8. Marks the dataset complete for each org unit and period.

This script is designed to be safe to rerun; it reuses existing data elements and
avoids creating duplicates when the same data element already exists in DHIS2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ee
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


def env_or_default(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def load_local_credentials(file_path: Optional[str] = None) -> Dict[str, str]:
    """Read credentials from a local text file such as file.txt.

    Supported formats include:
        USER_NAME="..."
        PASSWORD="..."
        URL="..."
        DATASET_UID="..."

    It also accepts uppercase/lowercase variants like USERNAME, DATASET_ID, DHIS2_URL.
    """
    values: Dict[str, str] = {}
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.txt")
    candidate_paths = [file_path, default_path] if file_path else [default_path]

    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = [part.strip() for part in line.split("=", 1)]
                    if not key:
                        continue
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    if key:
                        values[key] = value
        except OSError:
            continue

    normalized: Dict[str, str] = {}
    for key, value in values.items():
        normalized[key.lower()] = value

    resolved: Dict[str, str] = {}
    resolved["DHIS2_URL"] = (
        normalized.get("dhis2_url")
        or normalized.get("url")
        or env_or_default("DHIS2_URL", "https://hmis.mohcc.org.zw/nhis")
    ).strip().rstrip("/")
    resolved["DHIS2_USERNAME"] = (
        normalized.get("dhis2_username")
        or normalized.get("user_name")
        or normalized.get("username")
        or normalized.get("user")
        or env_or_default("DHIS2_USERNAME", "")
    )
    resolved["DHIS2_PASSWORD"] = (
        normalized.get("dhis2_password")
        or normalized.get("password")
        or env_or_default("DHIS2_PASSWORD", "")
    )
    resolved["DATASET_ID"] = (
        normalized.get("dataset_id")
        or normalized.get("dataset_uid")
        or normalized.get("dataset")
        or env_or_default("DATASET_ID", "qqND8igwWqz")
    )
    return resolved


LOCAL_CREDENTIALS = load_local_credentials()
DHIS2_URL = env_or_default("DHIS2_URL", LOCAL_CREDENTIALS["DHIS2_URL"]).strip().rstrip("/")
DHIS2_USERNAME = env_or_default("DHIS2_USERNAME", LOCAL_CREDENTIALS["DHIS2_USERNAME"])
DHIS2_PASSWORD = env_or_default("DHIS2_PASSWORD", LOCAL_CREDENTIALS["DHIS2_PASSWORD"])
DATASET_ID = env_or_default("DATASET_ID", LOCAL_CREDENTIALS["DATASET_ID"])
OUTPUT_DIR = env_or_default("OUTPUT_DIR", os.path.join(os.getcwd(), "dhis2_gee_prepared_data"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def show_progress(current: int, total: int, label: str = "Progress") -> None:
    if total <= 0:
        return
    pct = (current / total) * 100.0
    print(f"[{label}] {current}/{total} ({pct:.1f}%)")


def normalize_dhis2_code(label: str) -> str:
    code = (label or "").replace(" ", "_").replace("-", "_")
    code = code.replace("(°C)", "_C").replace("(%)", "_PCT")
    code = code.replace("(mm)", "_MM").replace("(W/m²)", "_W_M2")
    code = code.replace("(m/s)", "_M_S").replace("(°)", "_DEG")
    code = code.replace("/", "_").replace("(", "").replace(")", "")
    code = code.replace("°", "").replace("%", "PCT").replace("²", "2")
    code = code.replace("__", "_")
    return code.strip("_").upper()


class DHIS2Client:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, password)
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "DHIS2-GEE-Scheduler/1.0"
        })
        self.session.verify = True

    def request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        return self.session.request(method, url, timeout=(15, 300), **kwargs)

    def system_info(self) -> Dict[str, Any]:
        response = self.request("GET", "/api/system/info")
        response.raise_for_status()
        return response.json()

    def get_org_units(self, fields: str = "id,name,shortName,code,level,path,parent[id,name],geometry") -> List[Dict[str, Any]]:
        all_units: List[Dict[str, Any]] = []
        page = 1
        page_size = 500
        while True:
            response = self.request(
                "GET",
                "/api/organisationUnits.json",
                params={"fields": fields, "page": page, "pageSize": page_size},
            )
            response.raise_for_status()
            payload = response.json()
            units = payload.get("organisationUnits", [])
            if not units:
                break
            all_units.extend(units)
            if len(units) < page_size:
                break
            page += 1
        return all_units

    def get_data_elements(self) -> List[Dict[str, Any]]:
        response = self.request(
            "GET",
            "/api/dataElements",
            params={"paging": "false", "fields": "id,name,code"},
        )
        if response.status_code in (403, 404):
            return []
        response.raise_for_status()
        return response.json().get("dataElements", [])

    def dataset_exists(self, dataset_id: str) -> bool:
        if not dataset_id:
            return False
        response = self.request(
            "GET",
            f"/api/dataSets/{dataset_id}",
            params={"fields": "id,name"},
        )
        if response.status_code in (403, 404):
            log(f"Dataset metadata check for {dataset_id} returned {response.status_code}. Continuing because metadata access may be restricted.")
            return False
        response.raise_for_status()
        return bool(response.json().get("id"))

    def get_dataset_data_element_ids(self, dataset_id: str) -> set:
        if not dataset_id:
            return set()
        response = self.request(
            "GET",
            f"/api/dataSets/{dataset_id}/dataElements",
            params={"paging": "false", "fields": "id"},
        )
        if response.status_code in (403, 404):
            log(f"Dataset member list for {dataset_id} is not readable ({response.status_code}).")
            return set()
        response.raise_for_status()
        return {
            item.get("id")
            for item in response.json().get("dataElements", [])
            if item.get("id")
        }

    def find_data_element(self, name: str, code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        match_name = (name or "").strip()
        match_code = (code or "").strip()
        for element in self.get_data_elements():
            if match_name and (element.get("name") or "").strip() == match_name:
                return element
            if match_code and (element.get("code") or "").strip() == match_code:
                return element
        return None

    def create_data_element(self, name: str, code: str, short_name: Optional[str] = None) -> Optional[str]:
        existing = self.find_data_element(name, code)
        if existing is not None:
            return existing.get("id")

        payload = {
            "name": name,
            "shortName": short_name or (name[:50] if name else code[:50]),
            "code": code,
            "valueType": "NUMBER",
            "aggregationType": "AVERAGE",
            "domainType": "AGGREGATE",
            "zeroIsSignificant": False,
        }
        response = self.request("POST", "/api/dataElements", json=payload)
        if response.status_code in (400, 409):
            existing = self.find_data_element(name, code)
            if existing is not None:
                return existing.get("id")
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("response", {}).get("uid") or data.get("id")
        return None

    def ensure_data_element_in_dataset(self, dataset_id: str, data_element_id: str) -> bool:
        if not dataset_id or not data_element_id:
            return False
        existing_ids = self.get_dataset_data_element_ids(dataset_id)
        if data_element_id in existing_ids:
            return False
        response = self.request(
            "POST",
            f"/api/dataSets/{dataset_id}/dataElements/{data_element_id}",
        )
        if response.status_code in (403, 404, 409):
            log(f"Attachment for data element {data_element_id} to dataset {dataset_id} was not confirmed ({response.status_code}).")
            return False
        response.raise_for_status()
        return True

    def mark_dataset_complete(self, dataset_id: str, org_unit_id: str, period: str) -> bool:
        if not dataset_id or not org_unit_id or not period:
            return False
        response = self.request(
            "POST",
            "/api/dataSetCompleteness",
            params={"ds": dataset_id, "ou": org_unit_id, "pe": period},
        )
        if response.status_code in (200, 201, 202):
            return True
        if response.status_code in (403, 404):
            log(f"Dataset completion not confirmed for OU {org_unit_id} / period {period} ({response.status_code}).")
            return False
        response.raise_for_status()
        return False

    def upload_data_values(self, data_values: List[Dict[str, str]], chunk_size: int = 100) -> int:
        total = len(data_values)
        if total == 0:
            return 0

        uploaded = 0
        for start in range(0, total, chunk_size):
            chunk = data_values[start:start + chunk_size]
            response = self.request("POST", "/api/dataValueSets", json={"dataValues": chunk})
            if response.status_code not in (200, 201, 202):
                log(f"Chunk upload failed: {response.status_code} - {response.text[:500]}")
                response.raise_for_status()
            uploaded += len(chunk)
            show_progress(start + len(chunk), total, label="DHIS2 upload")
        return uploaded


def get_point_ous(org_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []
    for unit in org_units:
        geom = unit.get("geometry") or {}
        geom_type = geom.get("type")
        coords = geom.get("coordinates")
        if geom_type != "Point":
            continue
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError):
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        valid.append({
            "id": unit.get("id"),
            "name": unit.get("name"),
            "level": unit.get("level"),
            "longitude": lon,
            "latitude": lat,
            "path": unit.get("path"),
        })
    return valid


def prepare_orgunit_dataframe(point_orgunits: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(point_orgunits)
    if df.empty:
        return df
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    return df.sort_values(["level", "name"]).reset_index(drop=True)


def ee_initialize() -> None:
    try:
        ee.Initialize()
        log("Google Earth Engine is already initialized.")
    except Exception:
        log("Initializing Google Earth Engine for the first time...")
        ee.Authenticate()
        ee.Initialize()


def compute_heat_stress_proxy(temp_c: float, humidity_pct: float, wind_ms: float, radiation_wm2: float) -> float:
    temp_c = float(temp_c)
    humidity_pct = float(humidity_pct)
    wind_ms = max(float(wind_ms), 0.1)
    radiation_wm2 = float(radiation_wm2)
    return temp_c + (humidity_pct / 10.0) + (radiation_wm2 / 100.0) - (wind_ms * 0.5)


def compute_relative_humidity_pct(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    temp_c = np.asarray(temp_c, dtype=float)
    dewpoint_c = np.asarray(dewpoint_c, dtype=float)
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    rh = 100.0 * (e / es)
    return pd.Series(np.clip(rh, 0.0, 100.0))


def compute_wind_speed(u10: pd.Series, v10: pd.Series) -> pd.Series:
    return pd.Series(np.sqrt(np.square(u10.astype(float)) + np.square(v10.astype(float))))


def compute_wind_direction_deg(u10: pd.Series, v10: pd.Series) -> pd.Series:
    angle = np.degrees(np.arctan2(u10.astype(float), v10.astype(float)))
    return pd.Series((270.0 - angle) % 360.0)


def compute_wet_bulb_temperature(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp_c = np.asarray(temp_c, dtype=float)
    rh_pct = np.asarray(rh_pct, dtype=float)
    return pd.Series(
        temp_c * np.arctan(0.151977 * np.sqrt(rh_pct + 8.313659))
        + np.arctan(temp_c + rh_pct)
        - np.arctan(rh_pct - 1.676331)
        + 0.00391838 * np.power(rh_pct, 1.5) * np.arctan(0.023101 * rh_pct)
        - 4.686
    )


def compute_globe_temperature(temp_c: pd.Series, radiation_wm2: pd.Series) -> pd.Series:
    return pd.Series(temp_c.astype(float) + (0.02 * radiation_wm2.astype(float) / 10.0))


def convert_era5_radiation_to_wm2(value: float, resolution: str = "monthly", dates: Optional[pd.Series] = None) -> float:
    if pd.isna(value):
        return np.nan
    resolution_key = (resolution or "monthly").lower().strip()
    seconds_lookup = {
        "hourly": 3600.0,
        "h": 3600.0,
        "daily": 86400.0,
        "d": 86400.0,
        "weekly": 7 * 86400.0,
        "w": 7 * 86400.0,
        "monthly": 30 * 86400.0,
        "m": 30 * 86400.0,
        "yearly": 365 * 86400.0,
        "y": 365 * 86400.0,
        "ye": 365 * 86400.0,
        "me": 30 * 86400.0,
    }
    period_seconds = seconds_lookup.get(resolution_key, 3600.0)
    if dates is not None:
        date_series = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
        if not date_series.empty:
            if resolution_key in {"monthly", "me", "m"}:
                month_seconds = [pd.Timestamp(dt).daysinmonth * 86400.0 for dt in date_series]
                if month_seconds:
                    period_seconds = float(np.median(month_seconds))
            elif resolution_key in {"yearly", "y", "ye"}:
                year_seconds = [(366 if pd.Timestamp(dt).is_leap_year else 365) * 86400.0 for dt in date_series]
                if year_seconds:
                    period_seconds = float(np.median(year_seconds))
    return float(value) / max(period_seconds, 1.0)


def prepare_climate_columns(df: pd.DataFrame, resolution: str = "monthly") -> pd.DataFrame:
    out = df.copy()
    if "temperature_k" in out.columns:
        out["temperature_c"] = out["temperature_k"] - 273.15
    elif "temperature_kelvin" in out.columns:
        out["temperature_c"] = out["temperature_kelvin"] - 273.15

    if "precipitation_mm_raw" in out.columns:
        out["precipitation_mm"] = out["precipitation_mm_raw"] * 1000.0

    if "dewpoint_k" in out.columns:
        out["dewpoint_c"] = out["dewpoint_k"] - 273.15
    elif "dewpoint_temperature_2m" in out.columns:
        out["dewpoint_c"] = out["dewpoint_temperature_2m"] - 273.15

    if {"temperature_c", "dewpoint_c"}.issubset(out.columns):
        out["humidity_pct"] = compute_relative_humidity_pct(out["temperature_c"], out["dewpoint_c"])

    if {"u10", "v10"}.issubset(out.columns):
        out["wind_ms"] = compute_wind_speed(out["u10"], out["v10"])
        out["wind_direction_deg"] = compute_wind_direction_deg(out["u10"], out["v10"])

    if "surface_solar_radiation_downwards" in out.columns:
        out["radiation_wm2"] = out.apply(
            lambda row: convert_era5_radiation_to_wm2(
                row["surface_solar_radiation_downwards"],
                resolution=resolution,
                dates=out["date"] if "date" in out.columns else None,
            ),
            axis=1,
        )
    elif "radiation_wm2" in out.columns:
        out["radiation_wm2"] = out.apply(
            lambda row: convert_era5_radiation_to_wm2(
                row["radiation_wm2"],
                resolution=resolution,
                dates=out["date"] if "date" in out.columns else None,
            ),
            axis=1,
        )

    if {"temperature_c", "humidity_pct", "wind_ms", "radiation_wm2"}.issubset(out.columns):
        out["heat_stress_proxy"] = out.apply(
            lambda row: compute_heat_stress_proxy(
                row["temperature_c"],
                row["humidity_pct"],
                row["wind_ms"],
                row["radiation_wm2"],
            ),
            axis=1,
        )

    return out


def compute_thermal_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "temperature_c" in out.columns:
        out["Tdb"] = out["temperature_c"]
    if {"temperature_c", "humidity_pct"}.issubset(out.columns):
        out["Tw"] = compute_wet_bulb_temperature(out["temperature_c"], out["humidity_pct"])
    if {"temperature_c", "radiation_wm2"}.issubset(out.columns):
        out["Tg"] = compute_globe_temperature(out["temperature_c"], out["radiation_wm2"])
    if {"temperature_c", "dewpoint_c"}.issubset(out.columns):
        out["THI"] = out["temperature_c"] + 0.36 * out["dewpoint_c"] + 41.2
    if {"Tw", "Tg", "Tdb"}.issubset(out.columns):
        out["WBGT_out"] = 0.7 * out["Tw"] + 0.2 * out["Tg"] + 0.1 * out["Tdb"]
        out["WBGT_in"] = 0.7 * out["Tw"] + 0.3 * out["Tg"]
    return out


def resample_climate_dataframe(df: pd.DataFrame, resolution: str = "monthly") -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df.copy()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    rule_map = {
        "hourly": "h",
        "daily": "d",
        "weekly": "w",
        "monthly": "ME",
        "yearly": "YE",
    }
    rule = rule_map.get(str(resolution).lower(), "ME")

    ts = out.set_index("date")
    temporal_columns = [
        "temperature_c",
        "precipitation_mm",
        "humidity_pct",
        "wind_ms",
        "wind_direction_deg",
        "radiation_wm2",
        "dewpoint_c",
        "heat_stress_proxy",
        "WBGT_out",
        "WBGT_in",
        "THI",
        "Tw",
        "Tg",
        "Tdb",
    ]
    summary_frames = []
    for col in temporal_columns:
        if col not in ts.columns:
            continue
        stats = ts[col].resample(rule).agg(["mean", "min", "max"])
        stats.columns = [f"{col}_{stat}" for stat in stats.columns]
        summary_frames.append(stats)

    if summary_frames:
        summary_df = pd.concat(summary_frames, axis=1).reset_index()
        summary_df = summary_df.rename(columns={"index": "date"})
    else:
        summary_df = pd.DataFrame({"date": pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=rule)})

    metadata_cols = ["ou_id", "ou_name", "level", "longitude", "latitude"]
    metadata = out[metadata_cols].drop_duplicates().iloc[[0]].copy()
    for col in metadata_cols:
        if col in metadata.columns:
            summary_df[col] = metadata[col].iloc[0]
    return summary_df.sort_values("date").reset_index(drop=True)


def extract_single_ou_climate(ou_row: pd.Series, era5_collection, start_date: str, end_date: str, scale: int = 10000, resolution: str = "monthly") -> pd.DataFrame:
    geometry = ee.Geometry.Point([float(ou_row["longitude"]), float(ou_row["latitude"])])
    band_mapping = {
        "temperature_k": "temperature_2m",
        "precipitation_mm_raw": "total_precipitation",
        "u10": "u_component_of_wind_10m",
        "v10": "v_component_of_wind_10m",
        "radiation_wm2": "surface_solar_radiation_downwards",
        "dewpoint_k": "dewpoint_temperature_2m",
    }

    frames = []
    for output_name, band_name in band_mapping.items():
        images = era5_collection.filterDate(start_date, end_date).select(band_name)

        def extract_image(img, band_name=band_name, output_name=output_name, geometry=geometry):
            value = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                bestEffort=True,
            ).get(band_name)
            return ee.Feature(None, {
                "date": img.date().format("YYYY-MM-dd HH:mm"),
                output_name: value,
            })

        fc = ee.FeatureCollection(images.map(extract_image))
        payload = fc.getInfo()
        rows = [feature["properties"] for feature in payload.get("features", [])]
        if rows:
            frame = pd.DataFrame(rows)
            frame["date"] = pd.to_datetime(frame["date"])
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer")

    result["ou_id"] = ou_row["id"]
    result["ou_name"] = ou_row["name"]
    result["level"] = ou_row["level"]
    result["longitude"] = float(ou_row["longitude"])
    result["latitude"] = float(ou_row["latitude"])
    result = result.sort_values("date").reset_index(drop=True)
    result = prepare_climate_columns(result, resolution=resolution)
    result = compute_thermal_indicators(result)
    if resolution is not None:
        result = resample_climate_dataframe(result, resolution=resolution)
    return result


def build_climate_dataframe(orgunits_df: pd.DataFrame, era5_collection, start_date: str, end_date: str, resolution: str = "monthly", max_ous: Optional[int] = None, batch_size: int = 25) -> pd.DataFrame:
    working = orgunits_df.copy()
    if max_ous is not None:
        working = working.head(max_ous).copy()
    if working.empty:
        return pd.DataFrame()

    results = []
    total = len(working)
    for start in range(0, total, batch_size):
        batch = working.iloc[start:start + batch_size].copy()
        batch_no = (start // batch_size) + 1
        total_batches = max(1, (total + batch_size - 1) // batch_size)
        log(f"Processing batch {batch_no} of {total_batches} ({len(batch)} org units)")
        batch_results = []
        for idx, (_, ou_row) in enumerate(batch.iterrows(), start=1):
            show_progress(idx, len(batch), label=f"OU batch {batch_no}")
            try:
                result = extract_single_ou_climate(ou_row, era5_collection, start_date, end_date, scale=10000, resolution=resolution)
                if not result.empty:
                    batch_results.append(result)
            except Exception as exc:
                log(f"Could not extract climate for {ou_row.get('name')}: {type(exc).__name__}: {exc}")
        if batch_results:
            results.append(pd.concat(batch_results, ignore_index=True))

    if not results:
        return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values(["ou_id", "date"]).reset_index(drop=True)


def build_dhis2_column_map(final_export_df: pd.DataFrame) -> Dict[str, str]:
    map_by_output = {
        "Mean Temperature (°C)": "Air temperature (ERA5-Land)",
        "Min Temperature (°C)": "Min temperature (ERA5-Land)",
        "Max Temperature (°C)": "Max air temperature (ERA5-Land)",
        "Mean Precipitation (mm)": "Precipitation (ERA5-Land)",
        "Min Precipitation (mm)": "Min precipitation (ERA5-Land)",
        "Max Precipitation (mm)": "Max precipitation (ERA5-Land)",
        "Mean Humidity (%)": "Relative humidity (ERA5-Land)",
        "Min Humidity (%)": "Min humidity (ERA5-Land)",
        "Max Humidity (%)": "Max humidity (ERA5-Land)",
        "Mean Dew Point (°C)": "Dewpoint temperature (ERA5-Land)",
        "Min Dew Point (°C)": "Min dewpoint temperature (ERA5-Land)",
        "Max Dew Point (°C)": "Max dewpoint temperature (ERA5-Land)",
        "Mean Radiation (W/m²)": "Solar radiation (ERA5-Land)",
        "Min Radiation (W/m²)": "Min solar radiation (ERA5-Land)",
        "Max Radiation (W/m²)": "Max solar radiation (ERA5-Land)",
        "Mean Wind Speed (m/s)": "Wind speed (ERA5-Land)",
        "Min Wind Speed (m/s)": "Min wind speed (ERA5-Land)",
        "Max Wind Speed (m/s)": "Max wind speed (ERA5-Land)",
        "Mean Heat Stress Proxy": "Heat stress (ERA5-Land)",
        "Min Heat Stress Proxy": "Min heat stress (ERA5-Land)",
        "Max Heat Stress Proxy": "Max heat stress (ERA5-Land)",
    }
    active = {}
    for column, dhis2_name in map_by_output.items():
        if column in final_export_df.columns:
            active[column] = dhis2_name
    return active


def make_user_friendly_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "Date",
        "ou_id": "Org Unit ID",
        "ou_name": "Organisation Unit",
        "level": "Level",
        "longitude": "Longitude",
        "latitude": "Latitude",
        "temperature_c_mean": "Mean Temperature (°C)",
        "temperature_c_min": "Min Temperature (°C)",
        "temperature_c_max": "Max Temperature (°C)",
        "precipitation_mm_mean": "Mean Precipitation (mm)",
        "precipitation_mm_min": "Min Precipitation (mm)",
        "precipitation_mm_max": "Max Precipitation (mm)",
        "humidity_pct_mean": "Mean Humidity (%)",
        "humidity_pct_min": "Min Humidity (%)",
        "humidity_pct_max": "Max Humidity (%)",
        "wind_ms_mean": "Mean Wind Speed (m/s)",
        "wind_ms_min": "Min Wind Speed (m/s)",
        "wind_ms_max": "Max Wind Speed (m/s)",
        "wind_direction_deg": "Wind Direction (°)",
        "radiation_wm2_mean": "Mean Radiation (W/m²)",
        "radiation_wm2_min": "Min Radiation (W/m²)",
        "radiation_wm2_max": "Max Radiation (W/m²)",
        "dewpoint_c_mean": "Mean Dew Point (°C)",
        "dewpoint_c_min": "Min Dew Point (°C)",
        "dewpoint_c_max": "Max Dew Point (°C)",
        "heat_stress_proxy_mean": "Mean Heat Stress Proxy",
        "heat_stress_proxy_min": "Min Heat Stress Proxy",
        "heat_stress_proxy_max": "Max Heat Stress Proxy",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def dhis2_sync_climate_dataframe(dhis2_client: DHIS2Client, dataframe: pd.DataFrame, dataset_id: str, column_mapping: Dict[str, str], date_column: str = "Date", org_unit_column: str = "Org Unit ID", chunk_size: int = 100) -> List[Tuple[str, str, str]]:
    if dataframe.empty:
        log("No rows are available to push to DHIS2.")
        return []

    created: List[Tuple[str, str, str]] = []
    resolved: Dict[str, str] = {}
    total_setup = len(column_mapping)

    for idx, (column_name, dhis2_name) in enumerate(column_mapping.items(), start=1):
        if column_name not in dataframe.columns:
            continue
        show_progress(idx, total_setup, label="DHIS2 setup")
        element = dhis2_client.find_data_element(dhis2_name)
        if element is None:
            code = normalize_dhis2_code(dhis2_name)
            element_id = dhis2_client.create_data_element(dhis2_name, code, short_name=dhis2_name[:50])
        else:
            element_id = element.get("id")
        if element_id:
            dhis2_client.ensure_data_element_in_dataset(dataset_id, element_id)
            resolved[dhis2_name] = element_id
            created.append((column_name, dhis2_name, element_id))

    records: List[Dict[str, str]] = []
    for _, row in dataframe.iterrows():
        org_unit_id = str(row.get(org_unit_column, "")).strip()
        if not org_unit_id or org_unit_id.lower() in {"nan", "none"}:
            continue
        try:
            current_date = pd.to_datetime(row[date_column])
        except Exception:
            continue
        period = current_date.strftime("%Y%m%d")
        for column_name, dhis2_name in column_mapping.items():
            if column_name not in dataframe.columns:
                continue
            raw_value = row[column_name]
            if pd.isna(raw_value):
                continue
            element_id = resolved.get(dhis2_name)
            if not element_id:
                continue
            value = round(float(raw_value), 1)
            records.append({
                "dataElement": element_id,
                "orgUnit": org_unit_id,
                "period": period,
                "value": str(value),
            })

    if not records:
        log("No DHIS2 records prepared for upload.")
        return created

    total_records = len(records)
    for start in range(0, total_records, chunk_size):
        chunk = records[start:start + chunk_size]
        show_progress(start + len(chunk), total_records, label="DHIS2 upload")
        response = dhis2_client.request("POST", "/api/dataValueSets", json={"dataValues": chunk})
        if response.status_code not in (200, 201, 202):
            log(f"DHIS2 chunk upload failed: {response.status_code} - {response.text[:500]}")
            response.raise_for_status()
        for record in chunk:
            dhis2_client.mark_dataset_complete(dataset_id, record["orgUnit"], record["period"])

    log(f"Prepared {len(created)} DHIS2 data elements and uploaded {len(records)} DHIS2 values to dataset {dataset_id}.")
    return created


def build_date_window(start_date: Optional[str], end_date: Optional[str], months_back: Optional[int] = None) -> Tuple[str, str]:
    if start_date and end_date:
        return start_date, end_date

    today = datetime.today()
    if months_back is not None:
        end_dt = today.replace(day=1) - timedelta(days=1)
        start_dt = (end_dt.replace(day=1) - pd.DateOffset(months=months_back - 1)).replace(day=1)
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    # Default to the last completed month.
    end_dt = today.replace(day=1) - timedelta(days=1)
    start_dt = end_dt.replace(day=1)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract ERA5-Land climate data for DHIS2 org units and upload to DHIS2.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--dataset-id", default=DATASET_ID, help="DHIS2 dataset UID to receive the climate data.")
    parser.add_argument("--dhis2-url", default=DHIS2_URL, help="Base DHIS2 URL, for example https://<server>/<context>")
    parser.add_argument("--username", default=DHIS2_USERNAME, help="DHIS2 username.")
    parser.add_argument("--password", default=DHIS2_PASSWORD, help="DHIS2 password.")
    parser.add_argument("--max-ous", type=int, default=None, help="Optional limit to a number of org units for testing.")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of org units processed per batch.")
    parser.add_argument("--resolution", choices=["hourly", "daily", "weekly", "monthly", "yearly"], default="monthly", help="Temporal resolution to aggregate to.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder used to save exported CSV files.")
    parser.add_argument("--credentials-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.txt"), help="Local text file holding DHIS2 URL, username, password, and dataset UID.")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare and preview results without writing to DHIS2.")
    args = parser.parse_args()

    file_credentials = load_local_credentials(args.credentials_file)
    if not args.dhis2_url or args.dhis2_url == "https://hmis.mohcc.org.zw/nhis":
        args.dhis2_url = env_or_default("DHIS2_URL", file_credentials["DHIS2_URL"]).strip().rstrip("/")
    if not args.username:
        args.username = env_or_default("DHIS2_USERNAME", file_credentials["DHIS2_USERNAME"])
    if not args.password:
        args.password = env_or_default("DHIS2_PASSWORD", file_credentials["DHIS2_PASSWORD"])
    if not args.dataset_id or args.dataset_id == "qqND8igwWqz":
        args.dataset_id = env_or_default("DATASET_ID", file_credentials["DATASET_ID"])

    start_date, end_date = build_date_window(args.start_date, args.end_date)
    if args.start_date:
        start_date = args.start_date
    if args.end_date:
        end_date = args.end_date

    os.makedirs(args.output_dir, exist_ok=True)
    log(f"Using DHIS2 URL: {args.dhis2_url}")
    log(f"Date range: {start_date} to {end_date}")
    log(f"Dataset ID: {args.dataset_id}")

    client = DHIS2Client(args.dhis2_url, args.username, args.password)
    try:
        client.system_info()
    except Exception as exc:
        log(f"DHIS2 authentication failed: {exc}")
        return 1

    org_units = client.get_org_units()
    point_ous = prepare_orgunit_dataframe(get_point_ous(org_units))
    log(f"Valid point org units found: {len(point_ous)}")
    if point_ous.empty:
        log("No valid point org units were found. Nothing to process.")
        return 0

    ee_initialize()
    era5_collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

    climate_df = build_climate_dataframe(
        point_ous,
        era5_collection,
        start_date,
        end_date,
        resolution=args.resolution,
        max_ous=args.max_ous,
        batch_size=args.batch_size,
    )

    if climate_df.empty:
        log("No climate data extracted for the selected date range.")
        return 0

    export_df = make_user_friendly_climate_columns(climate_df.copy())
    csv_path = os.path.join(args.output_dir, "dhis2_gee_climate_schedule.csv")
    export_df.to_csv(csv_path, index=False)
    log(f"Climate export saved to: {csv_path}")

    if args.dry_run:
        log("Dry run complete — no DHIS2 upload was executed.")
        return 0

    dhis2_map = build_dhis2_column_map(export_df)
    if not dhis2_map:
        log("No DHIS2 column mapping was found in the exported climate output.")
        return 0

    log("DHIS2 mapping selected for upload:")
    for column, dhis2_name in dhis2_map.items():
        log(f"  - {column} -> {dhis2_name}")

    dhis2_sync_climate_dataframe(
        dhis2_client=client,
        dataframe=export_df,
        dataset_id=args.dataset_id,
        column_mapping=dhis2_map,
        date_column="Date",
        org_unit_column="Org Unit ID",
        chunk_size=100,
    )
    log("Climate upload process completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
