"""Download UCI Online Retail II and convert it to a single parquet file.

Usage:  python -m custintel.ingest

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii  (Chen, D., 2012), licence CC BY 4.0.
Real, messy transactions from a UK online gift retailer, 2009-12-01 to 2011-12-09 (~1.07M rows).
"""
from __future__ import annotations

import zipfile

import pandas as pd
import requests

from . import config as C

COLUMNS = {
    "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
    "Customer ID": "customer_id", "Country": "country",
}


def download(force: bool = False) -> None:
    xlsx = C.RAW_DIR / C.XLSX_NAME
    if xlsx.exists() and not force:
        return
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = C.RAW_DIR / "online_retail_ii.zip"
    with requests.get(C.DATA_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(C.RAW_DIR)


def to_parquet() -> pd.DataFrame:
    """Read both sheets of the workbook (slow, ~3-5 min) and save data/raw/transactions.parquet."""
    sheets = pd.read_excel(C.RAW_DIR / C.XLSX_NAME, sheet_name=None, dtype={"Invoice": str,
                           "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True).rename(columns=COLUMNS)
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["stock_code"] = df["stock_code"].astype(str)
    df["invoice"] = df["invoice"].astype(str)
    df["description"] = df["description"].astype("string")  # a few cells are numbers in the source
    df["country"] = df["country"].astype("string")
    df.to_parquet(C.RAW_DIR / "transactions.parquet", index=False)
    return df


def load_raw() -> pd.DataFrame:
    path = C.RAW_DIR / "transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m custintel.ingest` first")
    return pd.read_parquet(path)


if __name__ == "__main__":
    download()
    data = to_parquet()
    print(f"{len(data):,} rows, {data['invoice_date'].min()} -> {data['invoice_date'].max()}")
