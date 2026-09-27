"""Ücretsiz geçmiş veri katmanı: S&P 500 günlük OHLCV paneli + makro seriler + bilanço tarihleri.

* İlk çalıştırmada (veya panel yoksa) BACKFILL_YEARS yıllık geçmiş yfinance'ten indirilir.
* Her gün son INCREMENTAL_PERIOD penceresi çekilip panele eklenir.
* Örtüşen barlarda fiyat farkı > SPLIT_CHECK_TOL ise (bölünme / bedelsiz) o hissenin tüm
  geçmişi yeniden indirilir -> geçmiş her zaman bölünmeye göre düzeltilmiş kalır.
* Panel aylık gzip CSV parçaları halinde saklanır (data/ohlcv/ohlcv_YYYY_MM.csv.gz).
  Günlük commit yalnızca içinde bulunulan ayın küçük dosyasını değiştirir; repo şişmez.

Evren: Wikipedia S&P 500 bileşen listesi (ücretsiz). Evren yalnızca genişler; endeksten çıkan
hisselerin geçmişi korunur (survivorship bias azaltılır). Tickerlar Yahoo formatındadır (BRK.B -> BRK-B).

Yahoo 'Close' bölünmeye göre düzeltilmiştir, temettüye göre düzeltilmemiştir (auto_adjust=False).
Temettü günlerindeki küçük sapma etiket gürültüsü olarak kabul edilir ve belgelenmiştir.

Komut satırı:
    python price_history.py            # artımlı güncelleme (panel yoksa tam backfill)
    python price_history.py --full     # tam yeniden indirme + bileşen listesi + bilanço tarihleri
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

PANEL_COLS = ["tarih", "ticker", "open", "high", "low", "close", "volume"]

FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "LLY", "JPM", "V",
    "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "MRK", "CVX", "KO", "PEP", "ADBE",
    "CRM", "WMT", "BAC", "NFLX", "AMD", "TMO", "LIN", "MCD", "ACN", "CSCO", "ABT", "ORCL", "DHR",
    "WFC", "TXN", "INTU", "QCOM", "PM", "CAT", "GE", "IBM", "AMGN", "NOW", "ISRG", "GS", "SPGI",
    "UBER", "HON", "RTX", "BKNG", "AMAT", "LOW", "UNP", "PLD", "BLK", "MU", "LRCX", "PANW", "DE",
]


def norm_ticker(t):
    return str(t).upper().strip().replace(".", "-").replace("/", "-")


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------
def _yf():
    import yfinance as yf  # geç import: test ortamında sahte modül enjekte edilebilir
    return yf


def _ensure_dirs():
    os.makedirs(C.OHLCV_DIR, exist_ok=True)


def _read_universe_file():
    if os.path.exists(C.UNIVERSE_FILE):
        try:
            with open(C.UNIVERSE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def load_universe():
    tickers = [norm_ticker(t) for t in _read_universe_file().get("tickers", []) if t]
    return tickers or list(FALLBACK_UNIVERSE)


def load_sectors():
    return {norm_ticker(k): v for k, v in (_read_universe_file().get("sectors") or {}).items()}


def load_members():
    """Güncel endeks üyeleri (canlı taramada yalnızca bunlar işlem adayı olur)."""
    m = _read_universe_file().get("members")
    return set(norm_ticker(t) for t in m) if m else set(load_universe())


def save_universe(tickers, sectors=None, members=None):
    _ensure_dirs()
    old = _read_universe_file()
    tickers = sorted({norm_ticker(t) for t in tickers if t and str(t).strip()})
    sec = dict(old.get("sectors") or {})
    sec.update({norm_ticker(k): v for k, v in (sectors or {}).items() if v})
    mem = sorted(norm_ticker(t) for t in members) if members else old.get("members")
    with open(C.UNIVERSE_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now().strftime("%Y-%m-%d"), "tickers": tickers, "members": mem,
                   "sectors": sec}, f, ensure_ascii=False, indent=1)
    return tickers


def merge_universe(new_tickers, sectors=None, members=None):
    """Evren yalnızca genişler (endeksten çıkan hisseler geçmişte kalır -> survivorship azaltılır)."""
    current = set(_read_universe_file().get("tickers") or [])
    current.update(norm_ticker(t) for t in (new_tickers or []) if t)
    return save_universe(current, sectors=sectors, members=members)


def fetch_sp500_constituents():
    """Wikipedia'dan güncel S&P 500 bileşenleri + GICS sektörleri (ücretsiz). Hata olursa ({}, [])."""
    import io
    import requests
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0 (quant research bot)"}, timeout=20).text
        tables = pd.read_html(io.StringIO(html))
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if "symbol" in cols:
                t.columns = cols
                sec_col = next((c for c in cols if "gics sector" in c), None)
                tickers = [norm_ticker(x) for x in t["symbol"].dropna()]
                sectors = dict(zip(tickers, t[sec_col])) if sec_col else {}
                if len(tickers) > 400:
                    return sectors, tickers
    except Exception as exc:
        print(f"S&P 500 bileşen listesi alınamadı: {exc}")
    return {}, []


def refresh_constituents():
    sectors, members = fetch_sp500_constituents()
    if members:
        merge_universe(members, sectors=sectors, members=members)
        print(f"📋 S&P 500 bileşenleri güncellendi: {len(members)}")
    return members


# ------------------------------------------------------------------
# Panel saklama
# ------------------------------------------------------------------
def load_panel(min_date=None):
    files = sorted(glob.glob(os.path.join(C.OHLCV_DIR, "ohlcv_*.csv.gz")))
    if min_date is not None:
        min_date = pd.Timestamp(min_date)
        keep = []
        for f in files:
            try:
                y, m = os.path.basename(f).replace("ohlcv_", "").replace(".csv.gz", "").split("_")
                if pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0) >= min_date:
                    keep.append(f)
            except Exception:
                keep.append(f)
        files = keep
    if not files:
        return pd.DataFrame(columns=PANEL_COLS)
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            print(f"Panel parçası okunamadı {f}: {exc}")
    if not frames:
        return pd.DataFrame(columns=PANEL_COLS)
    df = pd.concat(frames, ignore_index=True)
    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce").dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["tarih", "ticker", "close"])
    df = df.drop_duplicates(["tarih", "ticker"], keep="last").sort_values(["tarih", "ticker"])
    if min_date is not None:
        df = df[df["tarih"] >= min_date]
    return df.reset_index(drop=True)


def save_panel(df, months=None):
    """Paneli aylık parçalara yazar. months verilirse yalnızca o aylar yeniden yazılır."""
    _ensure_dirs()
    if df is None or df.empty:
        return
    df = df[PANEL_COLS].copy()
    df["tarih"] = pd.to_datetime(df["tarih"]).dt.normalize()
    df = df.drop_duplicates(["tarih", "ticker"], keep="last").sort_values(["tarih", "ticker"])
    df["_ym"] = df["tarih"].dt.strftime("%Y_%m")
    targets = set(months) if months is not None else set(df["_ym"].unique())
    for ym, part in df.groupby("_ym"):
        if ym not in targets:
            continue
        out = part.drop(columns="_ym").copy()
        out["tarih"] = out["tarih"].dt.strftime("%Y-%m-%d")
        for c in ["open", "high", "low", "close"]:
            out[c] = out[c].round(4)
        out["volume"] = out["volume"].round(0)
        out.to_csv(os.path.join(C.OHLCV_DIR, f"ohlcv_{ym}.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})


# ------------------------------------------------------------------
# İndirme
# ------------------------------------------------------------------
def _extract_frame(data, symbol, single):
    """yfinance çıktısından tek sembolün OHLCV tablosunu güvenle çıkarır (sürüm farklarına dayanıklı)."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            lvl1 = data.columns.get_level_values(1)
            if symbol in lvl0:
                sub = data[symbol]
            elif symbol in lvl1:
                sub = data.xs(symbol, axis=1, level=1)
            else:
                return None
        else:
            if not single:
                return None
            sub = data
        sub = sub.rename(columns=lambda c: str(c).strip().lower())
        need = ["open", "high", "low", "close", "volume"]
        if any(c not in sub.columns for c in need):
            return None
        sub = sub[need].dropna(subset=["close"])
        if sub.empty:
            return None
        idx = pd.to_datetime(sub.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        sub = sub.copy()
        sub.index = idx.normalize()
        return sub
    except Exception:
        return None


def download_ohlcv(tickers, period=None, start=None, suffix=C.TICKER_SUFFIX, retries=2):
    """tickers -> uzun format panel (tarih, ticker, OHLCV). Sessizce başarısız olanları atlar."""
    yf = _yf()
    rows = []
    tickers = [t for t in tickers if t]
    for i in range(0, len(tickers), C.DOWNLOAD_CHUNK):
        chunk = tickers[i:i + C.DOWNLOAD_CHUNK]
        symbols = [f"{t}{suffix}" if suffix and not str(t).endswith(suffix) else t for t in chunk]
        data = None
        for attempt in range(retries + 1):
            try:
                kwargs = dict(interval="1d", group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
                if start is not None:
                    kwargs["start"] = start
                else:
                    kwargs["period"] = period or C.INCREMENTAL_PERIOD
                data = yf.download(symbols, **kwargs)
                if data is not None and not data.empty:
                    break
            except Exception as exc:
                print(f"yfinance parça hatası ({attempt + 1}): {exc}")
            time.sleep(2 + 3 * attempt)
        if data is None or data.empty:
            continue
        single = len(symbols) == 1
        for t, sym in zip(chunk, symbols):
            sub = _extract_frame(data, sym, single)
            if sub is None:
                continue
            sub = sub.reset_index().rename(columns={sub.index.name or "index": "tarih"})
            sub.columns = ["tarih", "open", "high", "low", "close", "volume"]
            sub["ticker"] = str(t).replace(suffix, "") if suffix else t
            rows.append(sub[PANEL_COLS])
        time.sleep(1.0)
    if not rows:
        return pd.DataFrame(columns=PANEL_COLS)
    out = pd.concat(rows, ignore_index=True)
    out["tarih"] = pd.to_datetime(out["tarih"]).dt.normalize()
    out = out[(out["close"] > 0) & (out["high"] >= out["low"])]
    return out


def backfill_full(tickers=None):
    tickers = tickers or load_universe()
    start = (pd.Timestamp.now() - pd.DateOffset(years=C.BACKFILL_YEARS)).strftime("%Y-%m-%d")
    print(f"📥 Tam backfill: {len(tickers)} hisse, başlangıç {start}")
    panel = download_ohlcv(tickers, start=start)
    if panel.empty:
        print("⚠️ Backfill verisi alınamadı.")
        return panel
    # Tam yeniden yazım: eski parçaları temizle
    for f in glob.glob(os.path.join(C.OHLCV_DIR, "ohlcv_*.csv.gz")):
        os.remove(f)
    save_panel(panel)
    print(f"✅ Backfill tamam: {panel['ticker'].nunique()} hisse, {panel['tarih'].nunique()} gün, {len(panel)} satır")
    return panel


def update_incremental(tickers=None):
    """Son pencereyi indirir, bölünme kontrolü yapar, paneli günceller. Güncel paneli döndürür."""
    panel = load_panel()
    tickers = tickers or load_universe()
    if panel.empty or panel["tarih"].nunique() < C.WF_MIN_TRAIN_DAYS // 2:
        backfill_full(tickers)
        return load_panel()

    fresh = download_ohlcv(tickers, period=C.INCREMENTAL_PERIOD)
    if fresh.empty:
        print("⚠️ Artımlı veri alınamadı; mevcut panel kullanılıyor.")
        return panel

    # Bölünme / düzeltme tespiti: örtüşen barlarda kapanış farkı
    merged = fresh.merge(panel[["tarih", "ticker", "close"]], on=["tarih", "ticker"], how="inner", suffixes=("", "_old"))
    merged["diff"] = (merged["close"] / merged["close_old"] - 1.0).abs()
    adjusted = sorted(merged.loc[merged["diff"] > C.SPLIT_CHECK_TOL, "ticker"].unique().tolist())

    # Yeni eklenen (panelde hiç olmayan) hisseler için tam geçmiş
    new_tickers = sorted(set(fresh["ticker"]) - set(panel["ticker"]))
    refetch = sorted(set(adjusted) | set(new_tickers))
    months_touched = set(fresh["tarih"].dt.strftime("%Y_%m"))

    if refetch:
        print(f"🔁 Tam geçmiş yeniden indiriliyor (bölünme/yeni): {len(refetch)} hisse")
        start = (pd.Timestamp.now() - pd.DateOffset(years=C.BACKFILL_YEARS)).strftime("%Y-%m-%d")
        hist = download_ohlcv(refetch, start=start)
        if not hist.empty:
            ok = set(hist["ticker"])
            panel = panel[~panel["ticker"].isin(ok)]
            panel = pd.concat([panel, hist], ignore_index=True)
            months_touched |= set(hist["tarih"].dt.strftime("%Y_%m"))

    panel = pd.concat([panel, fresh], ignore_index=True)
    panel = panel.drop_duplicates(["tarih", "ticker"], keep="last").sort_values(["tarih", "ticker"])
    save_panel(panel, months=months_touched)
    return panel.reset_index(drop=True)


def append_live_bar(panel, live_df, day=None):
    """Yahoo bugünün barını henüz yayınlamadıysa TradingView kesitini panele ekler (yalnızca eksikler)."""
    if live_df is None or live_df.empty:
        return panel
    day = pd.Timestamp(day or pd.Timestamp.now()).normalize()
    need = {"ticker", "open", "high", "low", "close", "volume"}
    if not need.issubset(live_df.columns):
        return panel
    live = live_df[list(need)].copy()
    live["tarih"] = day
    live = live[(live["close"] > 0) & (live["high"] >= live["low"]) & (live["volume"] > 0)]
    have = set(panel.loc[panel["tarih"] == day, "ticker"]) if not panel.empty else set()
    live = live[~live["ticker"].isin(have)]
    if live.empty:
        return panel
    out = pd.concat([panel, live[PANEL_COLS]], ignore_index=True)
    out = out.drop_duplicates(["tarih", "ticker"], keep="last").sort_values(["tarih", "ticker"]).reset_index(drop=True)
    save_panel(out, months={day.strftime("%Y_%m")})
    print(f"➕ Canlı kesitten {len(live)} hisse için {day.date()} barı eklendi.")
    return out


# ------------------------------------------------------------------
# Makro
# ------------------------------------------------------------------
def load_macro():
    if not os.path.exists(C.MACRO_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(C.MACRO_FILE, parse_dates=["tarih"])
        return df.set_index("tarih").sort_index()
    except Exception:
        return pd.DataFrame()


def update_macro(full=False):
    _ensure_dirs()
    old = load_macro()
    symbols = list(C.MACRO_TICKERS.values())
    if full or old.empty:
        start = (pd.Timestamp.now() - pd.DateOffset(years=C.BACKFILL_YEARS + 1)).strftime("%Y-%m-%d")
        raw = download_ohlcv(symbols, start=start, suffix="")
    else:
        raw = download_ohlcv(symbols, period="30d", suffix="")
    if raw.empty:
        print("⚠️ Makro veri alınamadı; eski makro kullanılıyor.")
        return old
    inv = {v: k for k, v in C.MACRO_TICKERS.items()}
    raw["name"] = raw["ticker"].map(inv)
    wide = raw.pivot_table(index="tarih", columns="name", values="close", aggfunc="last")
    if not old.empty:
        wide = wide.combine_first(old)  # yeni değerler öncelikli, eksikler eskiden
    wide = wide.sort_index()
    wide.index.name = "tarih"
    wide.reset_index().to_csv(C.MACRO_FILE, index=False, compression={"method": "gzip", "mtime": 0})
    return wide


# ------------------------------------------------------------------
# Bilanço tarihleri (geçmiş + planlanan) — backtest ve canlıda aynı karartma kuralı
# ------------------------------------------------------------------
def load_earnings():
    if not os.path.exists(C.EARNINGS_FILE):
        return pd.DataFrame(columns=["ticker", "date"])
    try:
        df = pd.read_csv(C.EARNINGS_FILE, parse_dates=["date"])
        df["date"] = df["date"].dt.normalize()
        return df.dropna()
    except Exception:
        return pd.DataFrame(columns=["ticker", "date"])


def update_earnings(tickers=None, limit=16):
    """yfinance get_earnings_dates ile ~4 yıllık geçmiş + gelecek bilanço tarihleri. En iyi çaba."""
    yf = _yf()
    tickers = tickers or load_universe()
    old = load_earnings()
    rows, ok = [], 0
    for i, t in enumerate(tickers):
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=limit)
            if ed is not None and len(ed):
                idx = pd.to_datetime(ed.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert("America/New_York").tz_localize(None)
                # Kapanış sonrası (>=16:00) açıklanan bilanço ertesi günün hareketini etkiler
                d = pd.Series(idx.normalize() + pd.to_timedelta((idx.hour >= 16).astype(int), unit="D"))
                rows.append(pd.DataFrame({"ticker": t, "date": d.values}))
                ok += 1
        except Exception:
            pass
        if i % 25 == 24:
            time.sleep(2.0)
    if rows:
        new = pd.concat(rows, ignore_index=True)
        allr = pd.concat([old, new], ignore_index=True)
        allr["date"] = pd.to_datetime(allr["date"]).dt.normalize()
        allr = allr.drop_duplicates().sort_values(["ticker", "date"])
        _ensure_dirs()
        allr.assign(date=allr["date"].dt.strftime("%Y-%m-%d")).to_csv(C.EARNINGS_FILE, index=False)
        print(f"📅 Bilanço tarihleri: {ok}/{len(tickers)} hisse")
        return allr
    print("⚠️ Bilanço tarihleri alınamadı; eski dosya kullanılıyor.")
    return old


def main(argv):
    full = "--full" in argv
    if full:
        refresh_constituents()
        backfill_full()
        update_earnings()
    else:
        update_incremental()
        if not os.path.exists(C.EARNINGS_FILE):
            update_earnings()
    update_macro(full=full)


if __name__ == "__main__":
    main(sys.argv[1:])
