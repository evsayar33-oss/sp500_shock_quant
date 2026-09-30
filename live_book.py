"""Canlı işlem defteri v3 + canlı kâr/zarar tabloları (5 günlük ve haftalık).

Defter satırı (label_version = 3) bir sinyaldir: T günü kapanışta üretilir, T+1 açılışta girilir.
Her tarama sonrası panelden KESİN işlem günleriyle yeniden hesaplanır (idempotent):
  giriş fiyatı, stop/TP1/TP2 seviyeleri, gün gün getiri (G1..G5), TP1 / başabaş / TP2 / stop / süre olayları,
  kapanışta net getiri (maliyet düşülmüş). Kural exits.py ile backtest etiketinin AYNISIDIR.

Tablolar (data/ altında, panel ve Telegram okur):
  live_trades.csv   : işlem başına 5 günlük tablo (G1..G5 kümülatif %, durum, sonuç, net)
  live_weekly.csv   : ISO haftası bazında kapanan işlemler (işlem, kazanma %, ort. net, portföy katkısı, TP1 %, stop %)
  live_summary.json : toplam / son 20 işlem özetleri + backtest beklentisiyle karşılaştırma
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
import exits

V = 3
TRADE_COLS = ["date", "ticker", "label_version", "signal_close", "weight_pct", "atr_pct", "initial_score",
              "effective_min_score", "p_win", "regime", "macro_label", "grp", "grp_name", "liq20", "cost_rt",
              "entry_date", "entry_price", "stop_price", "tp1_price", "tp2_price", "status", "exit_reason",
              "exit_date", "days", "tp1_hit", "tp1_date", "tp2_hit", "stop_hit", "mtm_ret", "net_ret",
              "d1", "d2", "d3", "d4", "d5", "is_completed"]
STATUS_TXT = {"PENDING": "Açılışta alınacak", "OPEN": "Açık", "TP1": "TP1 ✓ (stop girişte)", "CLOSED": "Kapandı"}


def _f(x, d=np.nan):
    try:
        x = float(x)
        return d if not np.isfinite(x) else x
    except Exception:
        return d


TEXT_COLS = ("date", "ticker", "entry_date", "exit_date", "tp1_date", "status", "exit_reason", "regime",
             "macro_label", "grp_name")


def _objcols(df: pd.DataFrame) -> pd.DataFrame:
    for c in TRADE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in TEXT_COLS:
        df[c] = df[c].astype(object)
    return df


def load_ledger() -> pd.DataFrame:
    if not os.path.exists(C.LEDGER_FILE):
        return pd.DataFrame(columns=TRADE_COLS)
    try:
        df = pd.read_csv(C.LEDGER_FILE)
    except Exception:
        return pd.DataFrame(columns=TRADE_COLS)
    return _objcols(df)


def v3_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    return ledger[pd.to_numeric(ledger["label_version"], errors="coerce") == V]


def record_signals(ledger: pd.DataFrame, port: pd.DataFrame, day) -> pd.DataFrame:
    """Portföye giren sinyalleri deftere ekler (aynı gün yeniden çalıştırmada o günün v3 satırları yenilenir)."""
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    if not ledger.empty:
        lv = pd.to_numeric(ledger["label_version"], errors="coerce")
        ledger = ledger[~((ledger["date"].astype(str) == day_str) & (lv == V))]
    if port is None or port.empty:
        return ledger
    picks = port[port["weight_pct"] > 0]
    rows = []
    for _, r in picks.iterrows():
        close = _f(r.get("close"))
        g = r.get("grp")
        rows.append({"date": day_str, "ticker": r["ticker"], "label_version": V, "signal_close": close,
                     "weight_pct": r["weight_pct"], "atr_pct": _f(r.get("atr")) / close if close else np.nan,
                     "initial_score": r.get("shock_score"), "effective_min_score": r.get("effective_min_score"),
                     "p_win": r.get("p_win"), "regime": r.get("meta_regime"), "macro_label": r.get("macro_label"),
                     "grp": g, "grp_name": r.get("grp_name"), "liq20": r.get("liq20"),
                     "cost_rt": float(C.round_trip_cost_pct(_f(r.get("liq20"), C.MIN_LIQ_TL))),
                     "status": "PENDING", "is_completed": 0})
    new = pd.DataFrame(rows, columns=TRADE_COLS)
    out = pd.concat([ledger, new], ignore_index=True, sort=False) if not ledger.empty else new
    return _objcols(out)


def update_book(ledger: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """v3 satırlarını panelden yeniden hesaplar. Dönüş: (defter, bugünkü olaylar listesi)."""
    events = []
    if ledger.empty or panel is None or panel.empty:
        return ledger, events
    ledger = _objcols(ledger.copy())
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(panel["tarih"].unique())))
    last_day = cal[-1]
    px = panel.set_index(["tarih", "ticker"])[["open", "high", "low", "close"]].sort_index()
    idx3 = v3_rows(ledger).index
    for i in idx3:
        r = ledger.loc[i]
        if int(_f(r.get("is_completed"), 0)) == 1 and pd.notna(r.get("net_ret")):
            continue
        t = r["ticker"]
        sig = pd.Timestamp(r["date"])
        pos = int(cal.searchsorted(sig, side="right"))
        if pos >= len(cal) or (cal[pos], t) not in px.index:
            ledger.at[i, "status"] = "PENDING"
            continue
        days = cal[pos:pos + C.HORIZON]
        bars = [px.loc[(d, t)] for d in days if (d, t) in px.index]
        if not bars:
            continue
        O = np.array([b["open"] for b in bars]); Hh = np.array([b["high"] for b in bars])
        L = np.array([b["low"] for b in bars]); Cc = np.array([b["close"] for b in bars])
        entry = float(O[0])
        atr_pct = _f(r.get("atr_pct"), 0.03)
        res = exits.run_path(O, Hh, L, Cc, entry, atr_pct)
        prev_status = str(r.get("status"))
        ledger.at[i, "entry_date"] = cal[pos].strftime("%Y-%m-%d")
        ledger.at[i, "entry_price"] = round(entry, 4)
        ledger.at[i, "stop_price"] = round(res["stop_level"], 4)
        ledger.at[i, "tp1_price"] = round(res["tp1_level"], 4)
        ledger.at[i, "tp2_price"] = round(res["tp2_level"], 4) if res["tp2_level"] else np.nan
        ledger.at[i, "days"] = res["days"]
        ledger.at[i, "tp1_hit"] = int(res["tp1"])
        ledger.at[i, "tp2_hit"] = int(res["tp2"])
        ledger.at[i, "stop_hit"] = int(res["stop"])
        if res["tp1_day"]:
            ledger.at[i, "tp1_date"] = days[res["tp1_day"] - 1].strftime("%Y-%m-%d")
        for k in range(C.HORIZON):
            ledger.at[i, f"d{k + 1}"] = round((Cc[k] / entry - 1) * 100.0, 2) if k < len(Cc) else np.nan
        ledger.at[i, "mtm_ret"] = round(res["mtm"] - _f(r.get("cost_rt"), 0.0), 3)
        name = r["ticker"]
        if res["closed"]:
            xd = days[res["exit_day"] - 1]
            net = round(res["gross"] - _f(r.get("cost_rt"), 0.0), 3)
            ledger.at[i, "status"] = "CLOSED"
            ledger.at[i, "exit_reason"] = res["reason"]
            ledger.at[i, "exit_date"] = xd.strftime("%Y-%m-%d")
            ledger.at[i, "net_ret"] = net
            ledger.at[i, "is_completed"] = 1
            if xd == last_day:
                events.append({"ticker": name, "event": res["reason"], "net": net})
        else:
            st = "TP1" if res["tp1"] else "OPEN"
            ledger.at[i, "status"] = st
            ledger.at[i, "is_completed"] = 0
            if res["tp1"] and res["tp1_day"] and days[res["tp1_day"] - 1] == last_day:
                events.append({"ticker": name, "event": "TP1", "net": None})
            if prev_status == "PENDING" and cal[pos] == last_day:
                events.append({"ticker": name, "event": "GİRİŞ", "net": None})
    return ledger, events


def positions_view(ledger: pd.DataFrame, panel: pd.DataFrame) -> list:
    """Telegram/panel için açık ve bekleyen pozisyonlar."""
    out = []
    v3 = v3_rows(ledger)
    if v3.empty:
        return out
    last = panel[panel["tarih"] == panel["tarih"].max()].set_index("ticker")["close"] if panel is not None and not panel.empty else pd.Series(dtype=float)
    for _, r in v3[v3["status"].isin(["PENDING", "OPEN", "TP1"])].iterrows():
        d = {"ticker": r["ticker"], "status": r["status"], "weight": _f(r.get("weight_pct"), 0.0)}
        if r["status"] == "PENDING":
            lv = exits.levels(_f(r.get("signal_close")), _f(r.get("atr_pct"), 0.03))
            d.update({"stop": lv["stop"], "tp1": lv["tp1"], "tp2": lv["tp2"], "approx": True})
        else:
            cur = _f(last.get(r["ticker"]))
            d.update({"held": int(_f(r.get("days"), 0)), "pnl": _f(r.get("mtm_ret"), 0.0),
                      "stop": _f(r.get("stop_price")), "tp1": _f(r.get("tp1_price")), "tp2": _f(r.get("tp2_price")),
                      "price": cur})
        out.append(d)
    return out


# ------------------------------------------------------------------
# Tablolar
# ------------------------------------------------------------------
def _metrics(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    g, l = s[s > 0].sum(), -s[s < 0].sum()
    return {"n": int(len(s)), "win_rate": round(float((s > 0).mean() * 100), 1), "avg_return": round(float(s.mean()), 3),
            "profit_factor": round(float(g / l), 2) if l > 0 else None, "sum": round(float(s.sum()), 2)}


def build_tables(ledger: pd.DataFrame, backtest: dict | None = None) -> dict:
    v3 = v3_rows(ledger).copy()
    os.makedirs(C.DATA_DIR, exist_ok=True)
    if v3.empty:
        summary = {"n_signals": 0, "all": {"n": 0}, "last20": {"n": 0}, "backtest": backtest or {}}
        json.dump(summary, open(os.path.join(C.DATA_DIR, "live_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        pd.DataFrame().to_csv(os.path.join(C.DATA_DIR, "live_trades.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(C.DATA_DIR, "live_weekly.csv"), index=False)
        return summary
    # 5 günlük tablo
    t = v3.sort_values(["date", "ticker"], ascending=[False, True])
    trades = pd.DataFrame({
        "Sinyal": t["date"], "Hisse": t["ticker"], "Ağırlık %": t["weight_pct"], "Giriş": t["entry_price"],
        **{f"G{k}": t[f"d{k}"] for k in range(1, C.HORIZON + 1)},
        "Durum": t["status"].map(STATUS_TXT).fillna(t["status"]), "Sonuç": t["exit_reason"],
        "Net %": t["net_ret"].where(t["status"] == "CLOSED", t["mtm_ret"]),
        "Kapanış": t["exit_date"], "Olasılık": t["p_win"]})
    trades.to_csv(os.path.join(C.DATA_DIR, "live_trades.csv"), index=False)
    # haftalık tablo (kapanış haftası)
    cl = v3[v3["status"] == "CLOSED"].copy()
    weekly = pd.DataFrame()
    if not cl.empty:
        cl["exit_dt"] = pd.to_datetime(cl["exit_date"])
        iso = cl["exit_dt"].dt.isocalendar()
        cl["Hafta"] = iso["year"].astype(str) + "-H" + iso["week"].astype(str).str.zfill(2)
        cl["net"] = pd.to_numeric(cl["net_ret"], errors="coerce")
        cl["katki"] = cl["net"] * pd.to_numeric(cl["weight_pct"], errors="coerce").fillna(0) / 100.0
        wk = cl.groupby("Hafta").agg(İşlem=("net", "size"), Kazanan=("net", lambda s: int((s > 0).sum())),
                                     **{"Ort. net %": ("net", "mean"), "Portföy katkısı %": ("katki", "sum"),
                                        "TP1 %": ("tp1_hit", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100),
                                        "Stop %": ("stop_hit", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100)})
        wk["Kazanma %"] = wk["Kazanan"] / wk["İşlem"] * 100
        wk = wk.sort_index()
        wk["Kümülatif katkı %"] = wk["Portföy katkısı %"].cumsum()
        weekly = wk.reset_index()[["Hafta", "İşlem", "Kazanan", "Kazanma %", "Ort. net %", "Portföy katkısı %",
                                   "Kümülatif katkı %", "TP1 %", "Stop %"]].round(2)
    weekly.to_csv(os.path.join(C.DATA_DIR, "live_weekly.csv"), index=False)
    closed = v3[v3["status"] == "CLOSED"].sort_values("exit_date")
    summary = {"n_signals": int(len(v3)), "open": int(v3["status"].isin(["OPEN", "TP1"]).sum()),
               "pending": int((v3["status"] == "PENDING").sum()),
               "all": _metrics(closed["net_ret"]), "last20": _metrics(closed["net_ret"].tail(20)),
               "tp1_rate": round(float(pd.to_numeric(closed["tp1_hit"], errors="coerce").mean() * 100), 1) if len(closed) else None,
               "stop_rate": round(float(pd.to_numeric(closed["stop_hit"], errors="coerce").mean() * 100), 1) if len(closed) else None,
               "since": str(v3["date"].min()), "backtest": backtest or {}}
    if not weekly.empty:
        summary["this_week"] = weekly.iloc[-1].to_dict()
    json.dump(summary, open(os.path.join(C.DATA_DIR, "live_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    return summary
