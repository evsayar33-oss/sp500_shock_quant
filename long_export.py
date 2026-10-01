"""Tek seferlik 10+ yıllık veri dışa aktarma (araştırma için). Repodaki 3 yıllık panele ve main dalına DOKUNMAZ.
Çıktı: long_out/ohlcv_YYYY.csv.gz (günlük OHLCV, sistemle aynı ayar: auto_adjust=False) + macro_long.csv.gz -> `long-data` dalı."""
import os, time
import pandas as pd
import config as C
H = __import__("bist_history" if os.path.exists("bist_history.py") else "price_history")
START = "2013-01-01"
OUT = "long_out"; os.makedirs(OUT, exist_ok=True)
uni = H.load_universe()
print(f"Evren: {len(uni)} hisse, başlangıç {START}")
panel = H.download_ohlcv(uni, start=START)
missing = sorted(set(uni) - set(panel["ticker"].unique())) if not panel.empty else uni
if missing:
    time.sleep(10)
    panel = pd.concat([panel, H.download_ohlcv(missing, start=START)], ignore_index=True)
panel = panel.drop_duplicates(["tarih", "ticker"]).sort_values(["tarih", "ticker"])
for y, g in panel.groupby(panel["tarih"].dt.year):
    g.to_csv(f"{OUT}/ohlcv_{y}.csv.gz", index=False, compression="gzip")
print(f"Panel: {panel['ticker'].nunique()} hisse, {panel['tarih'].min().date()} – {panel['tarih'].max().date()}, {len(panel)} satır")
raw = H.download_ohlcv(list(C.MACRO_TICKERS.values()), start=START, suffix="")
if not raw.empty:
    inv = {v: k for k, v in C.MACRO_TICKERS.items()}
    raw["name"] = raw["ticker"].map(inv)
    raw.pivot_table(index="tarih", columns="name", values="close", aggfunc="last").to_csv(f"{OUT}/macro_long.csv.gz", compression="gzip")
print("✅ tamam")
