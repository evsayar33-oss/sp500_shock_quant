"""Tek seferlik TEŞHİS + SİSTEM SAĞLIĞI dışa aktarımı (bist_shock_quant ve sp500_shock_quant için ortak).

Neden: Fiyat paneli zaten repoda (data/ohlcv) ve dışarıdan okunabiliyor. Dışarıdan GÖRÜLEMEYEN şey
GitHub Actions sunucusundaki çalışma zamanı gerçeğidir: veri kaynaklarına erişim, gecikme, panelin
Yahoo'nun bugünkü verisiyle tutarlılığı, bölünme/eksik gün şüpheleri, model ve defter durumu.

Çıktı: diag_out/diag_report.json + diag_out/diag_report.md  ->  ayrı `diag-data` dalına yazılır.
main dalına, verilere ve panele DOKUNMAZ. Hata olsa bile rapor üretir (her kontrol kendi try bloğunda).
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT = "diag_out"
os.makedirs(OUT, exist_ok=True)
R = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": {}}


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            res = fn()
            R["checks"][name] = {"ok": True, "sec": round(time.time() - t0, 1), **(res or {})}
        except Exception as exc:
            R["checks"][name] = {"ok": False, "sec": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc()[-1500:]}
        print(f"[{'OK ' if R['checks'][name]['ok'] else 'ERR'}] {name} ({R['checks'][name]['sec']}s)")
        return fn
    return deco


import config as C  # noqa: E402

IS_SP = "S&P" in getattr(C, "PROJECT_TITLE", "") or os.path.exists("sp_engine.py")
HIST = importlib.import_module(getattr(C, "HISTORY_MODULE", "price_history" if IS_SP else "bist_history"))
SUFFIX = getattr(C, "TICKER_SUFFIX", ".IS")
R["project"] = "sp500_shock_quant" if IS_SP else "bist_shock_quant"
SAMPLE = (["AAPL", "MSFT", "NVDA", "JPM", "XOM", "LLY", "BRK-B", "COST"] if IS_SP
          else ["THYAO", "GARAN", "ASELS", "EREGL", "BIMAS", "TUPRS", "KCHOL", "SISE"])


@check("ortam")
def _env():
    import yfinance
    return {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            "yfinance": getattr(yfinance, "__version__", "?"), "platform": platform.platform(),
            "tz_env": os.environ.get("TZ"), "local_now": datetime.now().isoformat(timespec="seconds")}


PANEL = {}


@check("panel_butunlugu")
def _panel():
    p = HIST.load_panel()
    PANEL["p"] = p
    p = p.copy()
    p["tarih"] = pd.to_datetime(p["tarih"])
    days = sorted(p["tarih"].unique())
    per_day = p.groupby("tarih").size()
    last30 = per_day.tail(30)
    c = p.pivot_table(index="tarih", columns="ticker", values="close").sort_index()
    ret = c.pct_change()
    big = (ret.abs() > getattr(C, "MAX_VALID_DAILY_MOVE", 0.25)).stack()
    big = big[big]
    last_seen = p.groupby("ticker")["tarih"].max()
    stale = last_seen[last_seen < days[-1] - pd.Timedelta(days=14)]
    bday_gaps = pd.Series(np.diff(np.array(days).astype("datetime64[D]")).astype(int))
    hl_bad = ((p["high"] < p[["open", "close"]].max(axis=1) - 1e-9) | (p["low"] > p[["open", "close"]].min(axis=1) + 1e-9))
    return {"ilk_gun": str(days[0])[:10], "son_gun": str(days[-1])[:10], "gun": len(days), "hisse": int(p["ticker"].nunique()),
            "satir": int(len(p)), "son_gun_kapsam": int(per_day.iloc[-1]), "son30_medyan_kapsam": float(last30.median()),
            "son30_min_kapsam": int(last30.min()), "duzeltilmemis_sicrama_supheli": int(len(big)),
            "sicrama_ornek": [f"{d.date()} {t}" for d, t in list(big.index)[-15:]],
            "bayat_hisse_14g": int(len(stale)), "bayat_ornek": list(stale.index[:15]),
            "en_uzun_takvim_boslugu_gun": int(bday_gaps.max()) if len(bday_gaps) else 0,
            "hl_tutarsiz_oran": float(hl_bad.mean()), "sifir_hacim_oran": float((p["volume"] <= 0).mean()),
            "yinelenen_satir": int(p.duplicated(["tarih", "ticker"]).sum())}


@check("yahoo_erisim_ve_tutarlilik")
def _yahoo():
    t0 = time.time()
    fresh = HIST.download_ohlcv(SAMPLE, period="15d")
    dl = round(time.time() - t0, 1)
    out = {"indirme_sn": dl, "istenen": len(SAMPLE), "gelen": int(fresh["ticker"].nunique()) if not fresh.empty else 0,
           "yahoo_son_gun": str(fresh["tarih"].max())[:10] if not fresh.empty else None}
    p = PANEL.get("p")
    if p is not None and not fresh.empty:
        m = fresh.merge(p[["tarih", "ticker", "open", "high", "low", "close"]], on=["tarih", "ticker"], suffixes=("", "_p"))
        for k in ("open", "high", "low", "close"):
            m[f"d_{k}"] = (m[k] / m[f"{k}_p"] - 1).abs()
        out["ortusen_bar"] = int(len(m))
        out["kapanis_fark_%>0.5"] = int((m["d_close"] > 0.005).sum())
        out["yuksek_dusuk_fark_%>0.5"] = int(((m["d_high"] > 0.005) | (m["d_low"] > 0.005)).sum())
        out["max_kapanis_fark_%"] = round(float(m["d_close"].max() * 100), 3) if len(m) else None
        out["fark_ornek"] = m.loc[m["d_close"] > 0.005, ["tarih", "ticker", "close", "close_p"]].astype(str).head(10).values.tolist()
    return out


@check("tradingview_erisim")
def _tv():
    mod = importlib.import_module("sp_fetcher" if IS_SP else "shock_fetcher")
    fn = getattr(mod, "get_sp500_raw_data" if IS_SP else "get_bist_raw_data")
    t0 = time.time()
    df = fn()
    return {"sn": round(time.time() - t0, 1), "satir": int(len(df)), "kolonlar": list(df.columns)[:15]}


@check("makro_seriler")
def _macro():
    m = HIST.load_macro()
    return {"seri": {c: str(m[c].dropna().index.max())[:10] for c in m.columns} if not m.empty else {},
            "satir": int(len(m))}


if IS_SP:
    @check("wikipedia_bilesenler")
    def _wiki():
        sec, mem = HIST.fetch_sp500_constituents()
        if not mem:
            raise RuntimeError("Wikipedia listesi boş döndü (erişim engeli veya sayfa biçimi değişti)")
        return {"uye": len(mem), "sektor": len(set(sec.values()))}

    @check("bilanco_tarihleri")
    def _earn():
        e = HIST.load_earnings()
        fut = e[e["date"] >= pd.Timestamp.now().normalize()]
        return {"satir": int(len(e)), "hisse": int(e["ticker"].nunique()) if len(e) else 0,
                "gelecek_tarihli_hisse": int(fut["ticker"].nunique()) if len(fut) else 0}


@check("model_durumu")
def _state():
    s = json.load(open(C.AI_STATE_FILE, encoding="utf-8"))
    me = s.get("meta_engine", {})
    return {"son_tarama": s.get("last_scan", {}).get("day"), "rejim": s.get("last_scan", {}).get("regime", {}).get("label"),
            "veri_kalitesi": s.get("last_scan", {}).get("data_quality"), "son_karar": me.get("last_decision"),
            "aktif_profil_var": bool(me.get("regime_profiles")), "terfi_sayisi": me.get("promotion_count"),
            "meta_filtre": (s.get("meta_label") or {}).get("enabled"), "wro": (s.get("win_rate_optimizer") or {}).get("status"),
            "otonomi": (s.get("autonomy_guard") or {}).get("mode"), "karne": s.get("backtest_summary")}


@check("defter_durumu")
def _ledger():
    L = pd.read_csv(C.LEDGER_FILE)
    lv = pd.to_numeric(L.get("label_version"), errors="coerce")
    v2 = L[lv == 2]
    done = pd.to_numeric(v2.get("net_ret_5d"), errors="coerce").dropna() if len(v2) else pd.Series(dtype=float)
    return {"toplam": int(len(L)), "eski_v1": int((lv != 2).sum()), "v2": int(len(v2)),
            "v2_tamamlanan": int(len(done)), "v2_kazanma_%": round(float((done > 0).mean() * 100), 1) if len(done) else None,
            "v2_net_ort": round(float(done.mean()), 3) if len(done) else None}


@check("repo_boyutu")
def _size():
    tot = 0
    for root, _, files in os.walk("."):
        if ".git" in root:
            continue
        tot += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return {"calisma_agaci_mb": round(tot / 1e6, 1)}


json.dump(R, open(f"{OUT}/diag_report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
with open(f"{OUT}/diag_report.md", "w", encoding="utf-8") as f:
    f.write(f"# Teşhis raporu — {R['project']} — {R['generated_utc']}\n\n")
    for k, v in R["checks"].items():
        f.write(f"## {'✅' if v['ok'] else '❌'} {k} ({v['sec']} sn)\n")
        for kk, vv in v.items():
            if kk not in ("ok", "sec", "trace"):
                f.write(f"- **{kk}**: {vv}\n")
        f.write("\n")
print("✅ teşhis tamam:", sum(v["ok"] for v in R["checks"].values()), "/", len(R["checks"]), "kontrol başarılı")
