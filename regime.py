"""Rejim katmanı: kesitsel (piyasa genişliği) + ABD makro (volatilite, kredi, trend, faiz, dolar).

Kesitsel sınıflandırıcı orijinal mantığı korur. Makro katman ücretsiz Yahoo serileriyle:
  * VIX seviyesi (1 yıllık yüzdelik) ve vade yapısı VIX/VIX3M (>1 = backwardation, akut stres)
  * SPY'nin 200 günlük ortalamaya göre trendi
  * Kredi: HYG - LQD 20g göreli getiri (kredi spread'i genişlemesi vekili)
  * Genişlik: RSP (eşit ağırlık) - SPY 20g göreli getiri (dar liderlik)
  * Faiz şoku: 10Y (^TNX) 20 günlük değişim; dolar şoku: DXY 20g getiri
Çıktılar: rejim yükseltme (VOL_SHOCK/RISK_OFF), eşik primi, brüt maruziyet çarpanı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

MACRO_FIELDS = ("vix", "vix_pct", "vix_term", "spy_trend200", "credit_rel20", "breadth_rel20", "tnx_chg20", "dxy_ret20")
NEUTRAL_MACRO = {"macro_label": "UNKNOWN", "macro_stress": 0.0, "macro_available": False,
                 **{k: np.nan for k in MACRO_FIELDS}}


# ------------------------------------------------------------------
# Kesitsel rejim (orijinal mantık)
# ------------------------------------------------------------------
def classify_market_regime(df):
    """Kesitsel rejim sınıflandırıcı. df['change_%'] (yüzde) kullanır."""
    base = {"label": "NORMAL", "confidence": 0.35, "is_crashing": False, "red_ratio": 0.50,
            "mean_change": 0.0, "median_change": 0.0, "dispersion": 0.0, "breadth_balance": 0.0}
    if df is None or df.empty or "change_%" not in df.columns:
        return dict(base)
    changes = pd.to_numeric(df["change_%"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if changes.empty:
        return dict(base)

    red_ratio = float((changes < 0).mean())
    green_ratio = float((changes > 0).mean())
    mean_change = float(changes.mean())
    median_change = float(changes.median())
    dispersion = float(changes.std(ddof=0)) if len(changes) > 1 else 0.0
    breadth_balance = green_ratio - red_ratio

    if red_ratio >= 0.70 or mean_change <= -1.50 or median_change <= -1.00:
        label = "CRASH"
    elif red_ratio >= 0.60 or median_change <= -0.45:
        label = "STRESS"
    elif green_ratio >= 0.62 and mean_change >= 0.55 and median_change >= 0.30:
        label = "EXPANSION"
    elif dispersion >= max(2.25, abs(mean_change) * 1.8 + 1.0) and 0.35 <= red_ratio <= 0.65:
        label = "ROTATION"
    elif dispersion <= 1.20 and abs(median_change) <= 0.35:
        label = "QUIET"
    else:
        label = "NORMAL"

    if label in ("CRASH", "STRESS"):
        terms = [red_ratio / 0.85, abs(min(median_change, 0.0)) / 1.80, abs(min(mean_change, 0.0)) / 2.50]
    elif label == "EXPANSION":
        terms = [green_ratio / 0.85, max(mean_change, 0.0) / 1.80, max(median_change, 0.0) / 1.20]
    elif label == "ROTATION":
        terms = [dispersion / 3.50, (1.0 - abs(red_ratio - 0.50) / 0.50)]
    else:
        terms = [1.0 - min(abs(median_change) / 1.5, 1.0), 1.0 - min(abs(breadth_balance) / 1.0, 1.0)]
    confidence = float(np.clip(np.mean(terms), 0.20, 0.98))

    return {"label": label, "confidence": round(confidence, 3), "is_crashing": label == "CRASH",
            "red_ratio": round(red_ratio, 4), "mean_change": round(mean_change, 3),
            "median_change": round(median_change, 3), "dispersion": round(dispersion, 3),
            "breadth_balance": round(breadth_balance, 4)}


# ------------------------------------------------------------------
# Makro katman
# ------------------------------------------------------------------
def _col(macro, name):
    return macro[name].astype(float).ffill() if name in macro.columns else None


def compute_macro_frame(macro: pd.DataFrame) -> pd.DataFrame:
    """Tarih indeksli makro stres tablosu. Her satır YALNIZCA o güne kadarki veriyi kullanır."""
    if macro is None or macro.empty:
        return pd.DataFrame()
    m = macro.sort_index()
    out = pd.DataFrame(index=m.index)
    comps = []
    ret20 = lambda s: (s / s.shift(20) - 1.0) * 100.0

    vix, vix3m = _col(m, "VIX"), _col(m, "VIX3M")
    if vix is not None:
        out["vix"] = vix
        out["vix_pct"] = vix.rolling(250, min_periods=60).rank(pct=True)
        s_vol = ((out["vix_pct"] - 0.60) / 0.40).clip(0, 1)
        if vix3m is not None:
            out["vix_term"] = vix / vix3m
            s_vol = np.maximum(s_vol, ((out["vix_term"] - 0.95) / 0.15).clip(0, 1))
        comps.append((0.30, s_vol))

    spy = _col(m, "SPY")
    if spy is not None:
        out["spy_trend200"] = (spy / spy.rolling(200, min_periods=100).mean() - 1.0) * 100.0
        comps.append((0.20, (-out["spy_trend200"] / 6.0).clip(0, 1)))

    hyg, lqd = _col(m, "HYG"), _col(m, "LQD")
    if hyg is not None and lqd is not None:
        out["credit_rel20"] = ret20(hyg) - ret20(lqd)
        comps.append((0.20, (-out["credit_rel20"] / 2.5).clip(0, 1)))

    rsp = _col(m, "RSP")
    if rsp is not None and spy is not None:
        out["breadth_rel20"] = ret20(rsp) - ret20(spy)
        comps.append((0.10, (-out["breadth_rel20"] / 3.0).clip(0, 1)))

    tnx = _col(m, "TNX")
    if tnx is not None:
        out["tnx_chg20"] = tnx - tnx.shift(20)          # yüzde puan (0.25 = 25bp)
        comps.append((0.10, ((out["tnx_chg20"] - 0.25) / 0.50).clip(0, 1)))

    dxy = _col(m, "DXY")
    if dxy is not None:
        out["dxy_ret20"] = ret20(dxy)
        comps.append((0.10, ((out["dxy_ret20"] - 1.5) / 3.0).clip(0, 1)))

    if not comps:
        return pd.DataFrame()
    wsum = sum(w for w, _ in comps)
    out["macro_stress"] = (sum(w * s.fillna(0.0) for w, s in comps) / wsum).clip(0, 1)

    label = pd.Series("NEUTRAL", index=out.index)
    trend = out.get("spy_trend200", pd.Series(0.0, index=out.index)).fillna(0.0)
    label[(out["macro_stress"] <= 0.20) & (trend > 0)] = "RISK_ON"
    label[out["macro_stress"] >= 0.55] = "RISK_OFF"
    if "vix_term" in out.columns:
        label[(out["vix_term"] >= 1.0) & (out.get("vix", 0) >= 22)] = "VOL_SHOCK"
    out["macro_label"] = label
    out["macro_available"] = True
    return out


def macro_snapshot(macro_frame: pd.DataFrame, day) -> dict:
    """Verilen gün (veya öncesindeki son gün) için makro durum."""
    if macro_frame is None or macro_frame.empty:
        return dict(NEUTRAL_MACRO)
    day = pd.Timestamp(day).normalize()
    sub = macro_frame[macro_frame.index <= day]
    if sub.empty:
        return dict(NEUTRAL_MACRO)
    row = sub.iloc[-1]
    # 7 günden eski makro veri güvenilmez sayılır
    if (day - sub.index[-1]).days > 7:
        return dict(NEUTRAL_MACRO)
    snap = dict(NEUTRAL_MACRO)
    for k in NEUTRAL_MACRO:
        if k in row.index:
            v = row[k]
            snap[k] = v if not isinstance(v, (float, np.floating)) or np.isfinite(v) else NEUTRAL_MACRO[k]
    snap["macro_stress"] = float(np.nan_to_num(snap.get("macro_stress", 0.0)))
    snap["macro_available"] = True
    return snap


def combine_regime(cross: dict, macro: dict) -> dict:
    """Kesitsel rejim + makro katman -> nihai rejim, eşik primi ve maruziyet çarpanı."""
    out = dict(cross)
    label = str(cross.get("label", "NORMAL")).upper()
    m_label = str(macro.get("macro_label", "UNKNOWN"))
    stress = float(np.clip(macro.get("macro_stress", 0.0) or 0.0, 0.0, 1.0))

    final = label
    if m_label == "VOL_SHOCK" and C.REGIME_SEVERITY.get(label, 2) < C.REGIME_SEVERITY["STRESS"]:
        final = "STRESS"
    elif m_label == "RISK_OFF" and label in ("EXPANSION", "QUIET", "NORMAL") and cross.get("median_change", 0.0) <= 0.0:
        final = "STRESS"

    out["cross_label"] = label
    out["label"] = final
    out["is_crashing"] = final == "CRASH"
    out["macro_label"] = m_label
    out["macro_stress"] = round(stress, 3)
    out["macro_threshold_add"] = round(6.0 * stress, 2)
    out["macro_exposure_mult"] = round(float(np.clip(1.0 - 0.6 * stress, 0.3, 1.0)), 3)
    for k in MACRO_FIELDS:
        v = macro.get(k, np.nan)
        try:
            out[k] = round(float(v), 3) if np.isfinite(float(v)) else None
        except Exception:
            out[k] = None
    return out
classify_bist_regime = classify_market_regime  # geriye uyumluluk
