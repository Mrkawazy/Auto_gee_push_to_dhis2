"""Mapping workbook parsing and the retrieval of a complete, cached input snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from . import dhis2
from .config import analysis_end_month, lag_rule

OPERAND_RE = re.compile(r"#?\{?\s*([A-Za-z][A-Za-z0-9]{10})(?:\.([A-Za-z][A-Za-z0-9]{10}))?\s*\}?")
KEY_TERMS = ["still", "live", "birth", "death", "test", "case", "deliver", "preterm", "weight", "complication"]


def parse_expression(expr) -> tuple[list[str], str]:
    """Return (operands, denominator_type) for a sum of DHIS2 operands; '1' or blank means no offset."""
    text = "" if expr is None or (isinstance(expr, float) and pd.isna(expr)) else str(expr).strip()
    if text in ("", "1", "1.0"):
        return [], "none_count_model"
    operands = [f"{m.group(1)}.{m.group(2)}" if m.group(2) else m.group(1) for m in OPERAND_RE.finditer(text)]
    leftover = OPERAND_RE.sub("", text).replace("+", "").replace("(", "").replace(")", "").strip()
    if leftover:
        raise ValueError(f"Only sums of DHIS2 operands are supported; cannot parse {text!r} (left: {leftover!r}).")
    if not operands:
        raise ValueError(f"No DHIS2 operands found in {text!r}.")
    return operands, "mapped_population"


def _pick_column(df: pd.DataFrame, must: str, prefer_uids: bool) -> str | None:
    cands = [c for c in df.columns if must in str(c).lower()]
    if not cands:
        return None
    if not prefer_uids:
        return cands[0]
    score = {c: df[c].astype(str).str.count(dhis2.UID_RE.pattern).sum() + df[c].astype(str).str.fullmatch(r"\s*1(\.0)?\s*").sum() for c in cands}
    return max(cands, key=lambda c: score[c])


def read_mapping_workbook(settings: dict) -> pd.DataFrame:
    """Locate the indicator, numerator and denominator columns and return one row per indicator."""
    path = Path(settings["workbook"])
    if not path.exists():
        raise FileNotFoundError(f"Mapping workbook not found: {path.resolve()}")
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    order = [settings["workbook_sheet"]] if settings.get("workbook_sheet") else list(sheets)
    explicit = settings.get("workbook_columns") or {}
    for name in order:
        raw = sheets[name]
        for header_row in range(min(15, len(raw))):
            cols = [str(c).strip() for c in raw.iloc[header_row].tolist()]
            if not any("numerator" in c.lower() for c in cols) and not explicit:
                continue
            df = raw.iloc[header_row + 1 :].copy()
            df.columns = cols
            df = df.dropna(how="all")
            num = explicit.get("numerator") or _pick_column(df, "numerator", True)
            den = explicit.get("denominator") or _pick_column(df, "denominator", True)
            label = explicit.get("name") or next(
                (c for c in df.columns if ("indicator" in c.lower() or "name" in c.lower()) and "numerator" not in c.lower() and "denominator" not in c.lower()),
                None,
            )
            if not label:
                label = next((c for alias in ("naming", "identification") for c in df.columns
                              if c.casefold() == alias), None)
            if num and den and label and all(c in df.columns for c in (num, den, label)):
                den_label = next((c for c in df.columns if "denominator" in c.lower() and c != den), None)
                out = pd.DataFrame(
                    {
                        "indicator_name": df[label].astype(str).str.strip(),
                        "source_numerator": df[num],
                        "source_denominator": df[den],
                        "denominator_label": df[den_label].astype(str) if den_label else "",
                    }
                )
                out = out.loc[out["source_numerator"].notna() & out["indicator_name"].ne("nan")].reset_index(drop=True)
                return out
    raise ValueError(
        f"Could not find indicator/numerator/denominator columns in {path.name}. "
        "Set settings['workbook_columns'] = {'name': ..., 'numerator': ..., 'denominator': ...}."
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24].strip("_")  # short: Windows paths are limited to 260 characters


def build_outcome_definitions(mapping: pd.DataFrame, settings: dict) -> list[dict]:
    """Outcome definitions with operands, lag windows, negative-control flag and parse errors."""
    outcomes = []
    nc_keys = [k.lower() for k in settings.get("negative_control_keywords", [])]
    for _, row in mapping.iterrows():
        rule = lag_rule(row["indicator_name"])
        error = ""
        try:
            num_ops, _ = parse_expression(row["source_numerator"])
            den_ops, den_type = parse_expression(row["source_denominator"])
        except ValueError as exc:
            num_ops, den_ops, den_type, error = [], [], "invalid", str(exc)
        digest = hashlib.sha1(f"{row['indicator_name']}|{row['source_numerator']}|{row['source_denominator']}".encode()).hexdigest()[:6]
        outcomes.append(
            {
                "outcome_id": f"{_slug(row['indicator_name'])}_{digest}",
                "indicator_name": row["indicator_name"],
                "source_numerator": str(row["source_numerator"]),
                "source_denominator": str(row["source_denominator"]),
                "denominator_label": row.get("denominator_label", ""),
                "numerator_operands": num_ops,
                "denominator_operands": den_ops,
                "denominator_type": den_type,
                "lag_max": rule["lag_max"],
                "alt_lag_max": rule["alt_lag_max"],
                "lag_rationale": rule["rationale"],
                "is_negative_control": any(k in row["indicator_name"].lower() for k in nc_keys),
                "parse_error": error,
            }
        )
    return outcomes


def mapping_warnings(outcomes: list[dict], names: dict[str, str]) -> None:
    """Flag labels whose key terms are absent from the mapped operands, and overlapping/odd mappings."""
    for o in outcomes:
        warnings = []
        if o["parse_error"]:
            warnings.append(o["parse_error"])
        missing = [op for op in o["numerator_operands"] + o["denominator_operands"] if op.split(".")[0] not in names]
        if missing:
            warnings.append(f"operands not found in DHIS2: {', '.join(missing)}")
        den_names = " ".join(names.get(op.split(".")[0], "") for op in o["denominator_operands"]).lower()
        label = str(o.get("denominator_label") or "").lower()
        if o["denominator_operands"] and label and label != "nan":
            absent = [t for t in KEY_TERMS if t in label and t not in den_names]
            if absent:
                warnings.append(f"denominator label mentions {absent} but mapped operands do not")
        if o["denominator_type"] == "mapped_population" and set(o["numerator_operands"]) == set(o["denominator_operands"]):
            warnings.append("numerator and denominator are identical")
        o["mapping_warning"] = "; ".join(warnings)
        o["numerator_names"] = "; ".join(names.get(op.split(".")[0], op) for op in o["numerator_operands"])
        o["denominator_names"] = "; ".join(names.get(op.split(".")[0], op) for op in o["denominator_operands"]) or "none (count model)"


def mapping_validation_table(outcomes: list[dict]) -> pd.DataFrame:
    cols = ["outcome_id", "indicator_name", "source_numerator", "numerator_names", "source_denominator", "denominator_names",
            "denominator_type", "lag_max", "alt_lag_max", "lag_rationale", "is_negative_control", "mapping_warning"]
    return pd.DataFrame(outcomes)[cols]


def _snapshot_key(settings: dict, workbook_hash: str) -> str:
    keys = {k: settings[k] for k in ["input_dataset", "start_month", "district_level", "province_level", "confounders"]}
    keys["end"] = str(analysis_end_month(settings))
    keys["workbook"] = workbook_hash
    return hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_inputs(client, settings: dict, refresh: bool = True) -> dict:
    """Retrieve (or reload) facilities, climate variables, outcome definitions and all values."""
    folder = Path(settings["output_folder"]) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    mapping = read_mapping_workbook(settings)
    wb_hash = hashlib.sha1(Path(settings["workbook"]).read_bytes()).hexdigest()[:12]
    key = _snapshot_key(settings, wb_hash)
    meta_path = folder / "snapshot.json"
    if not refresh and meta_path.exists() and json.loads(meta_path.read_text())["key"] == key:
        meta = json.loads(meta_path.read_text())
        print(f"Reusing saved input snapshot {key} retrieved {meta['retrieved']}")
        return {
            "facilities": pd.read_parquet(folder / "facilities.parquet"),
            "climate_vars": pd.read_parquet(folder / "climate_vars.parquet"),
            "outcomes": meta["outcomes"],
            "operand_datasets": meta["operand_datasets"],
            "values": pd.read_parquet(folder / "values.parquet"),
            "reports": pd.read_parquet(folder / "reports.parquet"),
            "snapshot_key": key,
            "snapshot_hash": meta["data_hash"],
            "retrieved": meta["retrieved"],
        }
    if not refresh:
        print("No matching saved snapshot; retrieving from DHIS2.")

    facilities, climate_vars = dhis2.fetch_climate_metadata(client, settings)
    if facilities.empty:
        raise RuntimeError("No point facilities are assigned to the climate dataset.")
    outcomes = build_outcome_definitions(mapping, settings)
    conf_ops = {name: dhis2_ops for name, expr in settings.get("confounders", {}).items() for dhis2_ops in [parse_expression(expr)[0]]}
    health_ops = sorted({op for o in outcomes for op in o["numerator_operands"] + o["denominator_operands"]} | {op for ops in conf_ops.values() for op in ops})
    de_uids = sorted({op.split(".")[0] for op in health_ops})
    names = dhis2.fetch_names(client, de_uids)
    mapping_warnings(outcomes, names)
    # An unknown UID makes DHIS2 reject the whole analytics batch, so only existing operands are requested;
    # outcomes with unknown operands are flagged above and end up unreported (never silently zero).
    health_ops = [op for op in health_ops if op.split(".")[0] in names]
    operand_datasets = dhis2.fetch_operand_datasets(client, de_uids)
    print(f"Metadata: {len(climate_vars)} climate variables; {len(outcomes)} workbook outcomes; {len(facilities)} point facilities")

    end = analysis_end_month(settings)
    start = pd.Period(settings["start_month"], freq="M")
    max_lag = max([o["alt_lag_max"] for o in outcomes] + [0])
    ous = facilities["facility_id"].tolist()
    health = dhis2.fetch_analytics(client, health_ops, dhis2.month_periods(start, end), ous, "health")
    climate = dhis2.fetch_analytics(client, climate_vars["climate_uid"].tolist(), dhis2.month_periods(start - max_lag, end + settings["forecast_horizon"]), ous, "climate")
    datasets = sorted({ds for v in operand_datasets.values() for ds in v})
    try:
        reports = dhis2.fetch_analytics(client, [f"{ds}.ACTUAL_REPORTS" for ds in datasets], dhis2.month_periods(start, end), ous, "reporting")
    except Exception as exc:  # without reporting records, blank components stay missing (strict mode)
        print(f"WARNING: reporting records unavailable ({str(exc)[:200]}); blank components will be treated as missing.")
        reports = pd.DataFrame(columns=["dx", "period", "ou", "value"])
    reports = reports.assign(dataset=reports["dx"].str.split(".").str[0])[["dataset", "period", "ou", "value"]] if not reports.empty else pd.DataFrame(columns=["dataset", "period", "ou", "value"])
    values = pd.concat([health.assign(kind="health"), climate.assign(kind="climate")], ignore_index=True)
    values["period"] = values["period"].astype(str)
    reports["period"] = reports["period"].astype(str)

    data_hash = hashlib.sha1(pd.util.hash_pandas_object(values[["dx", "period", "ou", "value"]], index=False).values.tobytes()).hexdigest()[:12]
    retrieved = pd.Timestamp.now().isoformat(timespec="seconds")
    facilities.to_parquet(folder / "facilities.parquet")
    climate_vars.to_parquet(folder / "climate_vars.parquet")
    values.to_parquet(folder / "values.parquet")
    reports.to_parquet(folder / "reports.parquet")
    meta_path.write_text(json.dumps({"key": key, "retrieved": retrieved, "data_hash": data_hash, "outcomes": outcomes, "operand_datasets": operand_datasets}, indent=1))
    print(f"Retrieved {len(health)} health values, {len(climate)} climate values, {len(reports)} reporting records")
    return {
        "facilities": facilities,
        "climate_vars": climate_vars,
        "outcomes": outcomes,
        "operand_datasets": operand_datasets,
        "values": values,
        "reports": reports,
        "snapshot_key": key,
        "snapshot_hash": data_hash,
        "retrieved": retrieved,
    }
