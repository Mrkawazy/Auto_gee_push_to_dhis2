"""DHIS2 Web API access: credentials, metadata, analytics retrieval and imports."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

UID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{10}")


def read_credentials(path: str | Path = "file.txt") -> dict:
    """Read the URL, username and password from environment variables, then `file.txt`.

    Accepted file layouts: `key=value` / `key: value` lines (url, username/user, password)
    or three plain lines in the order url, username, password.
    """
    creds = {
        "url": os.environ.get("DHIS2_URL") or os.environ.get("DHIS_NHIS_URL"),
        "username": os.environ.get("DHIS2_USERNAME") or os.environ.get("DHIS2_USER") or os.environ.get("nhis_user"),
        "password": os.environ.get("DHIS2_PASSWORD") or os.environ.get("nhis_pswr"),

    }
    if all(creds.values()):
        return creds
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "DHIS2 credentials not found: set DHIS2_URL, DHIS2_USER and DHIS2_PASSWORD "
            f"or create {path.resolve()} with url, username and password."
        )
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    parsed = {}
    for ln in lines:
        m = re.match(r"^\s*([A-Za-z_ ]+?)\s*[:=]\s*(.+)$", ln)
        if m and not ln.lower().startswith("http"):
            parsed[m.group(1).strip().lower().replace(" ", "_")] = m.group(2).strip().strip("'\"")
    if parsed:
        creds["url"] = creds["url"] or parsed.get("url") or parsed.get("dhis2_url") or parsed.get("base_url")
        creds["username"] = creds["username"] or parsed.get("username") or parsed.get("user_name") or parsed.get("user") or parsed.get("dhis2_username") or parsed.get("dhis2_user")
        creds["password"] = creds["password"] or parsed.get("password") or parsed.get("dhis2_password")
    elif len(lines) >= 3:
        creds["url"], creds["username"], creds["password"] = creds["url"] or lines[0], creds["username"] or lines[1], creds["password"] or lines[2]
    if not all(creds.values()):
        raise ValueError(f"{path} must provide url, username and password.")
    return creds


class DHIS2Client:
    """Small authenticated HTTPS session around the DHIS2 Web API with retries."""

    def __init__(self, url: str, username: str, password: str, timeout: int = 300):
        url = url.rstrip("/")
        self.base = url[: -len("/api")] if url.endswith("/api") else url
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.session.headers.update({"Accept": "application/json"})
        self.timeout = timeout

    def _request(self, method: str, endpoint: str, retries: int | None = None, **kwargs) -> dict:
        if retries is None:
            retries = 4 if method in {"GET", "DELETE"} or endpoint in {"dataValueSets", "completeDataSetRegistrations"} else 1
        url = f"{self.base}/api/{endpoint.lstrip('/')}"
        for attempt in range(retries):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 401:
                raise PermissionError("DHIS2 rejected the credentials (HTTP 401).")
            if resp.status_code >= 400 and resp.status_code != 409:
                raise RuntimeError(f"DHIS2 {method} {endpoint} failed: HTTP {resp.status_code} {resp.text[:500]}")
            body = resp.json() if resp.content else {}
            if isinstance(body, dict) and resp.headers.get("Location"):
                body["location"] = resp.headers["Location"]
            return body
        return {}

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, json: dict, params: dict | None = None) -> dict:
        return self._request("POST", endpoint, json=json, params=params)

    def post_once(self, endpoint: str, json: dict, params: dict | None = None) -> dict:
        return self._request("POST", endpoint, retries=1, json=json, params=params)

    def delete(self, endpoint: str, params: dict | None = None) -> dict:
        return self._request("DELETE", endpoint, params=params)


def make_client(credentials_path: str | Path = "file.txt"):
    """Create the DHIS2 client. `GHAP_MOCK_DHIS2=1` swaps in the synthetic test server."""
    if os.environ.get("GHAP_MOCK_DHIS2") == "1":
        from .mock import MockDHIS2Client

        return MockDHIS2Client()
    creds = read_credentials(credentials_path)
    return DHIS2Client(creds["url"], creds["username"], creds["password"])


# ---------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------


def _point(geometry) -> tuple[float, float]:
    """Longitude/latitude of a point, or the mean vertex of a polygon."""
    if not geometry or "coordinates" not in geometry:
        return float("nan"), float("nan")
    coords = geometry["coordinates"]
    if geometry.get("type") == "Point":
        return float(coords[0]), float(coords[1])
    flat = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            flat.append(c)
        else:
            for cc in c:
                walk(cc)

    walk(coords)
    return (sum(p[0] for p in flat) / len(flat), sum(p[1] for p in flat) / len(flat)) if flat else (float("nan"), float("nan"))


def fetch_climate_metadata(client, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (facilities, climate_variables) for the climate dataset."""
    ds = client.get(
        f"dataSets/{settings['input_dataset']}",
        {"fields": "id,name,dataSetElements[dataElement[id,name,shortName,code,aggregationType]],"
                   "organisationUnits[id,name,level,geometry,ancestors[id,name,level]]"},
    )
    climate = pd.DataFrame(
        [
            {
                "climate_uid": e["dataElement"]["id"],
                "climate_name": e["dataElement"].get("name", e["dataElement"]["id"]),
                "climate_short": e["dataElement"].get("shortName") or e["dataElement"].get("name", ""),
                "aggregation_type": e["dataElement"].get("aggregationType", ""),
            }
            for e in ds.get("dataSetElements", [])
        ]
    ).sort_values("climate_name", ignore_index=True)
    rows = []
    for ou in ds.get("organisationUnits", []):
        geom = ou.get("geometry") or {}
        lon, lat = _point(geom)
        anc = {a.get("level"): a for a in ou.get("ancestors", [])}
        district = anc.get(settings["district_level"], {})
        province = anc.get(settings["province_level"], {})
        national = anc.get(1, {})
        rows.append(
            {
                "facility_id": ou["id"],
                "facility_name": ou.get("name", ou["id"]),
                "level": ou.get("level"),
                "geometry_type": geom.get("type", "none"),
                "lon": lon,
                "lat": lat,
                "district_id": district.get("id", "unknown_district"),
                "district_name": district.get("name", "Unknown district"),
                "province_id": province.get("id", "unknown_province"),
                "province_name": province.get("name", "Unknown province"),
                "national_id": national.get("id", "unknown_national"),
                "national_name": national.get("name", "National"),
            }
        )
    facilities = pd.DataFrame(rows)
    facilities = facilities.loc[facilities["geometry_type"].eq("Point")].sort_values("facility_id", ignore_index=True)
    return facilities, climate


def fetch_operand_datasets(client, de_uids: list[str]) -> dict[str, list[str]]:
    """Map each data element to the datasets that collect it (used for zero-fill of submitted forms)."""
    out: dict[str, list[str]] = {}
    for i in range(0, len(de_uids), 50):
        chunk = de_uids[i : i + 50]
        resp = client.get(
            "dataElements",
            {"filter": f"id:in:[{','.join(chunk)}]", "fields": "id,name,dataSetElements[dataSet[id]]", "paging": "false"},
        )
        for de in resp.get("dataElements", []):
            out[de["id"]] = sorted({e["dataSet"]["id"] for e in de.get("dataSetElements", [])})
    return out


def fetch_names(client, uids: list[str]) -> dict[str, str]:
    """Display names for data elements (used in the mapping validation table)."""
    names: dict[str, str] = {}
    for i in range(0, len(uids), 50):
        chunk = uids[i : i + 50]
        resp = client.get("dataElements", {"filter": f"id:in:[{','.join(chunk)}]", "fields": "id,name", "paging": "false"})
        names.update({d["id"]: d["name"] for d in resp.get("dataElements", [])})
    return names


# ---------------------------------------------------------------------------------
# Analytics retrieval
# ---------------------------------------------------------------------------------


def month_periods(start: pd.Period, end: pd.Period) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def fetch_analytics(client, dx: list[str], periods: list[str], org_units: list[str], label: str = "", max_cells: int = 40000) -> pd.DataFrame:
    """Retrieve dx x pe x ou values from the analytics API in batches; absent values are simply missing."""
    if not dx:
        return pd.DataFrame(columns=["dx", "period", "ou", "value"])
    frames = []
    dx_chunk = max(1, min(20, max_cells // max(1, len(org_units) * min(len(periods), 24))))
    pe_chunk = max(1, min(len(periods), 24))
    ou_chunk = 100
    total = ((len(dx) - 1) // dx_chunk + 1) * ((len(periods) - 1) // pe_chunk + 1) * ((len(org_units) - 1) // ou_chunk + 1)
    done = 0
    for i in range(0, len(dx), dx_chunk):
        for j in range(0, len(periods), pe_chunk):
            for k in range(0, len(org_units), ou_chunk):
                params = [
                    ("dimension", "dx:" + ";".join(dx[i : i + dx_chunk])),
                    ("dimension", "pe:" + ";".join(periods[j : j + pe_chunk])),
                    ("dimension", "ou:" + ";".join(org_units[k : k + ou_chunk])),
                    ("skipMeta", "true"),
                    ("outputIdScheme", "UID"),
                ]
                resp = client.get("analytics.json", params)
                if resp.get("status") == "ERROR" or resp.get("httpStatusCode", 200) >= 400:
                    raise RuntimeError(f"DHIS2 analytics rejected a {label} batch: {resp.get('message', resp)}")
                headers = [h["name"] for h in resp.get("headers", [])]
                rows = resp.get("rows", [])
                if rows:
                    df = pd.DataFrame(rows, columns=headers)
                    frames.append(df.rename(columns={"pe": "period"})[["dx", "period", "ou", "value"]])
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"Retrieved {label} batch {done} / {total}")
    if not frames:
        return pd.DataFrame(columns=["dx", "period", "ou", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])
