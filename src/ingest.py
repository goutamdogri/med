from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
TARGET = RAW / "salesdaily.csv"
BACKUP = RAW / "salesdaily_original_backup.csv"

ATC_CODES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
DATE_FORMATS = ["%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d"]


def parse_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="raise")
            if parsed.notna().all():
                return parsed
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(s, errors="coerce", dayfirst=False)


def detect_format(df: pd.DataFrame) -> str:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    has_date = any(k in cols_lower for k in ("datum", "date", "day", "ds"))
    id_col = next(
        (
            cols_lower[k]
            for k in ("atc_code", "atc", "sku_id", "sku", "category", "drug")
            if k in cols_lower
        ),
        None,
    )
    units_col = next(
        (cols_lower[k] for k in ("units", "qty", "quantity", "sales", "y") if k in cols_lower),
        None,
    )
    if has_date and id_col and units_col:
        return "long"
    atc_present = [c for c in df.columns if str(c).upper().strip() in ATC_CODES]
    if has_date and len(atc_present) >= 2:
        return "wide"
    return "unknown"


def normalize_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    date_col = next(
        c for c in df.columns if c.lower().strip() in ("datum", "date", "day", "ds")
    )
    atc_cols = [c for c in df.columns if str(c).upper().strip() in ATC_CODES]
    out = pd.DataFrame({"date": parse_dates(df[date_col])})
    warnings = []
    for c in atc_cols:
        code = str(c).upper().strip()
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
        neg = int((vals < 0).sum())
        if neg:
            warnings.append(f"{code}: {neg} negative values clipped to 0")
        out[code] = vals.clip(lower=0).round(2)
    unknown = [
        c
        for c in df.columns
        if c not in atc_cols
        and c != date_col
        and pd.api.types.is_numeric_dtype(df[c])
        and str(c).upper().strip() not in ATC_CODES
    ]
    if unknown:
        warnings.append(f"ignored numeric columns: {', '.join(map(str, unknown))}")
    return out.groupby("date").sum(min_count=1).reset_index(), warnings


def normalize_long(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    date_col = next(c for c in cols_lower.values() if c.lower() in ("datum", "date", "day", "ds"))
    id_col = next(
        (
            cols_lower[k]
            for k in ("atc_code", "atc", "sku_id", "sku", "category", "drug")
            if k in cols_lower
        ),
        None,
    )
    units_col = next(
        (cols_lower[k] for k in ("units", "qty", "quantity", "sales", "y") if k in cols_lower),
        None,
    )
    out = pd.DataFrame(
        {
            "date": parse_dates(df[date_col]),
            "id": df[id_col].astype(str).str.upper().str.strip(),
            "units": pd.to_numeric(df[units_col], errors="coerce").fillna(0).clip(lower=0),
        }
    )
    warnings = []
    unmapped = sorted(set(out["id"]) - set(ATC_CODES))
    sku_like = [i for i in unmapped if "-" in i]
    if unmapped and not sku_like:
        raise ValueError(f"unrecognized category ids: {unmapped[:5]}")
    if sku_like:
        warnings.append(f"mapped {len(sku_like)} SKU-level ids to their ATC prefix")
        out["id"] = out["id"].str.split("-").str[0]
    out["id"] = out["id"].where(out["id"].isin(ATC_CODES), other=pd.NA)
    dropped = int(out["id"].isna().sum())
    if dropped:
        warnings.append(f"dropped {dropped} rows with unknown categories")
    out = out.dropna(subset=["id"])
    wide = out.pivot_table(index="date", columns="id", values="units", aggfunc="sum")
    wide = wide.reindex(columns=ATC_CODES).dropna(how="all", axis=1).fillna(0)
    return wide.reset_index(), warnings


def load_any_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    fmt = detect_format(df)
    if fmt == "long":
        data, warns = normalize_long(df)
    elif fmt == "wide":
        data, warns = normalize_wide(df)
    else:
        raise ValueError("could not detect date + ATC columns; check headers")

    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    full_idx = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    missing = len(full_idx) - len(data)

    report = {
        "format": fmt,
        "rows": len(data),
        "date_min": str(data["date"].min().date()),
        "date_max": str(data["date"].max().date()),
        "categories": [c for c in data.columns if c != "date"],
        "missing_days": max(missing, 0),
        "warnings": warns,
        "ok": True,
    }
    absent = sorted(set(ATC_CODES) - set(report["categories"]))
    if absent:
        report["warnings"].append(f"missing required categories: {', '.join(absent)}")
        report["ok"] = False
    if len(data) < 400:
        report["warnings"].append("less than ~13 months of history: year-lag features degrade")
        report["ok"] = False
    if data["date"].dt.dayofweek.nunique() < 7:
        report["warnings"].append("calendar coverage incomplete")
    return data, report


def install(csv_path: Path) -> dict:
    data, report = load_any_csv(csv_path)
    if TARGET.exists() and not BACKUP.exists():
        shutil.copy(TARGET, BACKUP)
        report["warnings"].append("original dataset backed up to salesdaily_original_backup.csv")
    data.to_csv(TARGET, index=False, date_format="%m/%d/%Y")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    args = ap.parse_args()
    report = install(args.csv_path)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
