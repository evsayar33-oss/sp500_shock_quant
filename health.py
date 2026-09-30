"""Sistem Sağlığı — her taramada hesaplanır; Telegram'da tek satır, panelde ayrıntılı sekme.

Kontroller (teşhis raporunda gerçekten görülen sorunlara göre):
  Veri    : son bar gecikmesi, son gün kapsamı, son barın kaynağı (Yahoo / TradingView yedeği),
            geçersiz (düzeltilmemiş) sıçramalar, makro serilerin güncelliği
  Kaynak  : Yahoo artımlı indirme, TradingView erişimi
  Model   : son denetimin yaşı, backtest örneklemi, otonomi modu
  Canlı   : canlı kazanma oranı backtest beklentisinden istatistiksel olarak sapıyor mu (binom z),
            olasılık filtresinin kalibrasyonu (tahmin edilen vs gerçekleşen)
  Çalışma : tarama / denetim / haftalık yenileme en son ne zaman çalıştı (heartbeat)
Her kontrol: ok / warn / bad. Skor = 100 − 12×warn − 30×bad (0..100).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

ICON = {"ok": "🟢", "warn": "🟡", "bad": "🔴"}


def _hours_since(ts):
    try:
        return (datetime.now() - pd.Timestamp(ts).tz_localize(None).to_pydatetime()).total_seconds() / 3600.0
    except Exception:
        return None


def heartbeat(state: dict, key: str):
    state.setdefault("heartbeat", {})[key] = datetime.now().isoformat(timespec="seconds")


def evaluate(state: dict, panel: pd.DataFrame, macro: pd.DataFrame | None, ctx: dict) -> dict:
    """ctx: live_rows (TV satırı), tv_appended (int), yahoo_rows (int), dq, guard_mode, live_summary."""
    items = []

    def add(group, name, level, detail):
        items.append({"group": group, "name": name, "level": level, "detail": detail})

    # --- Veri ---
    if panel is not None and not panel.empty:
        last = pd.Timestamp(panel["tarih"].max())
        today = pd.Timestamp(datetime.now().date())
        lag = int(np.busday_count(last.date(), today.date()))
        after_close = datetime.now().hour >= int(getattr(C, "HEALTH_CLOSE_HOUR", 18))
        exp_lag = 0 if after_close else 1
        add("Veri", "Son bar", "ok" if lag <= exp_lag else ("warn" if lag <= exp_lag + 1 else "bad"),
            f"son gün {last.date()} ({lag} iş günü önce)")
        per_day = panel.groupby("tarih").size()
        cov = per_day.iloc[-1] / max(per_day.tail(21).iloc[:-1].median(), 1) if len(per_day) > 2 else 1.0
        add("Veri", "Son gün kapsamı", "ok" if cov >= 0.95 else ("warn" if cov >= 0.8 else "bad"),
            f"{int(per_day.iloc[-1])} hisse (%{cov * 100:.0f})")
        tv_n = int(ctx.get("tv_appended") or 0)
        add("Veri", "Son bar kaynağı", "ok" if tv_n < per_day.iloc[-1] * 0.5 else "warn",
            f"Yahoo {int(per_day.iloc[-1]) - tv_n} · TradingView yedeği {tv_n}")
        c = panel[panel["tarih"] >= panel["tarih"].max() - pd.Timedelta(days=10)].pivot_table(
            index="tarih", columns="ticker", values="close").sort_index()
        jumps = int((c.pct_change().iloc[-1].abs() > C.MAX_VALID_DAILY_MOVE).sum()) if len(c) > 1 else 0
        add("Veri", "Geçersiz sıçrama (bugün)", "ok" if jumps == 0 else "warn",
            f"{jumps} hisse (düzeltilmemiş kurumsal işlem şüphesi, işlem dışı)")
    else:
        add("Veri", "Panel", "bad", "panel yok")

    if macro is not None and not macro.empty:
        lags = {}
        for col in macro.columns:
            s = macro[col].dropna()
            if len(s):
                lags[col] = int(np.busday_count(pd.Timestamp(s.index.max()).date(), datetime.now().date()))
        stale = {k: v for k, v in lags.items() if v > 2}
        add("Veri", "Makro seriler", "ok" if not stale else "warn",
            "hepsi güncel" if not stale else "gecikmeli: " + ", ".join(f"{k}({v}g)" for k, v in stale.items()))

    # --- Kaynaklar ---
    yl = ctx.get("yahoo_last")
    ylag = int(np.busday_count(pd.Timestamp(yl).date(), datetime.now().date())) if yl else 99
    add("Kaynak", "Yahoo (artımlı)", "ok" if ylag <= 1 else ("warn" if ylag <= 2 else "bad"),
        f"Yahoo son gün {yl} · {max(int(ctx.get('yahoo_rows') or 0), 0)} hisse" + (" (bugünkü bar TradingView'den)" if ylag >= 1 and (ctx.get('tv_appended') or 0) > 0 else ""))
    add("Kaynak", "TradingView", "ok" if (ctx.get("live_rows") or 0) > 10 else "warn", f"{ctx.get('live_rows', 0)} satır")

    # --- Model ---
    me = state.get("meta_engine", {})
    ts = (me.get("last_decision") or {}).get("timestamp")
    h = _hours_since(ts) if ts else None
    add("Model", "Son yeniden eğitim", "ok" if h is not None and h <= 30 else ("warn" if h is not None and h <= 80 else "bad"),
        f"{h:.0f} saat önce" if h is not None else "hiç çalışmadı")
    bt = state.get("backtest_summary") or {}
    add("Model", "Backtest örneklemi", "ok" if (bt.get("n") or 0) >= 150 else "warn",
        f"{bt.get('n', 0)} işlem · kazanma %{bt.get('win_rate', 0)} · net %{bt.get('avg_return', 0):+}")
    gm = ctx.get("guard_mode") or (state.get("autonomy_guard") or {}).get("mode", "NORMAL")
    add("Model", "Otonomi modu", {"NORMAL": "ok", "WATCH": "warn", "RECOVERY": "warn"}.get(gm, "bad"), gm)
    rt = state.get("retrain") or {}
    if rt.get("pending"):
        add("Model", "Yeniden eğitim tetikleyicisi", "warn", ", ".join(rt.get("reasons", [])))

    # --- Canlı performans ---
    ls = ctx.get("live_summary") or {}
    lm = ls.get("all") or {}
    n = lm.get("n", 0)
    if n >= 20 and bt.get("win_rate"):
        p0 = float(bt["win_rate"]) / 100.0
        wr = float(lm["win_rate"]) / 100.0
        z = (wr - p0) / math.sqrt(max(p0 * (1 - p0), 1e-6) / n)
        add("Canlı", "Kazanma oranı vs backtest", "ok" if z > -1 else ("warn" if z > -2 else "bad"),
            f"canlı %{lm['win_rate']} ({n} işlem) · beklenen %{bt['win_rate']} · z={z:+.1f}")
        add("Canlı", "Net getiri", "ok" if lm.get("avg_return", 0) > 0 else "warn",
            f"işlem başı %{lm.get('avg_return', 0):+.2f} · PF {lm.get('profit_factor')}")
    else:
        add("Canlı", "Canlı sonuç", "ok", f"{n} kapanmış işlem (anlamlı karşılaştırma için ≥20 gerekir)")
    cal = ctx.get("calibration")
    if cal:
        add("Canlı", "Olasılık kalibrasyonu", "ok" if abs(cal["gap"]) <= 12 else "warn",
            f"tahmin %{cal['pred']:.0f} · gerçekleşen %{cal['real']:.0f} ({cal['n']} işlem)")

    # --- Çalışma (heartbeat) ---
    hb = state.get("heartbeat") or {}
    for key, name, limit in (("scan", "Günlük tarama", 30), ("audit", "Öğrenme denetimi", 30), ("weekly", "Haftalık yenileme", 200)):
        hh = _hours_since(hb.get(key)) if hb.get(key) else None
        lvl = "ok" if hh is not None and hh <= limit else ("warn" if hh is None or hh <= limit * 2.5 else "bad")
        add("Çalışma", name, lvl, f"{hh:.0f} saat önce" if hh is not None else "kayıt yok")

    n_warn = sum(i["level"] == "warn" for i in items)
    n_bad = sum(i["level"] == "bad" for i in items)
    score = max(0, 100 - 12 * n_warn - 30 * n_bad)
    status = "bad" if n_bad else ("warn" if n_warn else "ok")
    out = {"ts": datetime.now().isoformat(timespec="seconds"), "score": score, "status": status,
           "icon": ICON[status], "items": items}
    try:
        os.makedirs(C.DATA_DIR, exist_ok=True)
        json.dump(out, open(os.path.join(C.DATA_DIR, "health.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass
    state["health"] = {"score": score, "status": status, "ts": out["ts"],
                       "issues": [f"{i['name']}: {i['detail']}" for i in items if i["level"] != "ok"]}
    return out


def calibration(ledger: pd.DataFrame) -> dict | None:
    """Olasılık filtresi açıksa: kapanmış işlemlerde ortalama tahmin vs gerçekleşen kazanma."""
    if ledger is None or ledger.empty or "p_win" not in ledger.columns:
        return None
    d = ledger[(ledger.get("status") == "CLOSED")]
    d = d[pd.to_numeric(d["p_win"], errors="coerce").notna()]
    if len(d) < 20:
        return None
    pred = float(pd.to_numeric(d["p_win"]).mean() * 100)
    real = float((pd.to_numeric(d["net_ret"], errors="coerce") > 0).mean() * 100)
    return {"pred": pred, "real": real, "gap": real - pred, "n": int(len(d))}


# ------------------------------------------------------------------
# Otomatik yeniden eğitim tetikleyicileri
# ------------------------------------------------------------------
def retrain_triggers(state: dict, regime_label: str, guard: dict, live_summary: dict) -> dict:
    """Canlı veriye göre 'güçlü yeniden eğitim' gerekip gerekmediğine karar verir.
    Güçlü eğitim = denetimde tam-geçmiş adayına ek olarak YAKIN DÖNEM (kayan pencere) adayı da denenir."""
    rt = state.setdefault("retrain", {"pending": False, "reasons": [], "history": []})
    reasons = []
    # 1) Canlı performans backtest beklentisinin belirgin altında
    bt = state.get("backtest_summary") or {}
    l20 = (live_summary or {}).get("last20") or {}
    if l20.get("n", 0) >= 20 and bt.get("win_rate"):
        if l20["win_rate"] < float(bt["win_rate"]) - 12 or (l20.get("avg_return", 0) < 0 and (l20.get("profit_factor") or 1) < 0.8):
            reasons.append(f"CANLI_ZAYIF (son 20: %{l20['win_rate']} / net %{l20.get('avg_return', 0):+.2f})")
    # 2) Kalıcı rejim değişimi (3 tarama boyunca yeni rejim)
    hist = state.setdefault("regime_hist", [])
    hist.append(regime_label)
    del hist[:-10]
    if len(hist) >= 6 and len(set(hist[-3:])) == 1 and hist[-4] != hist[-1] and hist[-5] != hist[-1]:
        reasons.append(f"REJİM_DEĞİŞTİ ({hist[-4]} → {hist[-1]})")
    # 3) Özellik dağılımı kayması
    if float(guard.get("drift_score", 0) or 0) >= 0.45:
        reasons.append(f"VERİ_KAYMASI (drift {guard.get('drift_score'):.2f})")
    if reasons:
        rt["pending"] = True
        rt["reasons"] = sorted(set(rt.get("reasons", []) + reasons))
        rt["history"] = (rt.get("history", []) + [{"ts": datetime.now().isoformat(timespec="seconds"), "reasons": reasons}])[-30:]
    return rt
