"""S&P 500 Adaptive Meta-Engine v2 — skor katmanı.

Bu modüldeki `score_frame` hem canlı taramada hem walk-forward backtest'te hem de öğrenmede
KULLANILAN TEK skor fonksiyonudur. Canlı skor ile geçmiş doğrulama arasında formül farkı yoktur.

Mimari
  1) features.py : gerçek z-skorlar, OHLCV birikim vekili, gap-riski, kesitsel aileler
  2) regime.py   : kesitsel + makro rejim, eşik primi, maruziyet çarpanı
  3) score_frame : rejime koşullu aile ağırlıkları (öğrenilmiş profil ⊕ rejim şablonu)
  4) portfolio.py: volatilite hedefli boyut, korelasyon filtresi, brüt limit
Fiyat yönü ana skoru üretmez; yalnızca uygunluk kapısıdır (change_% > 0).

v2.1: 6. aile "sector" (sektör/grup akışı, sector_flow.py) + İŞARETLİ ağırlıklar.
Öğrenici, OOS'ta istatistiksel olarak NEGATİF IC veren bir aileyi ters çevirebilir
(profil["signs"][aile] = -1 -> skor katkısı 100 - aile_skoru). Örn. BIST'te gerçek veride
"olay" ve "aktivite" (gürültülü hacim şokları) T+5'te geri dönüş öngörüyor (t≈-5).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
from regime import classify_market_regime  # noqa: F401  (geri uyumluluk için yeniden dışa aktarım)

GECMIS_DOSYA = C.GECMIS_DOSYA

DEFAULT_META_WEIGHTS = {"event": 0.17, "flow": 0.21, "activity": 0.17, "liquidity": 0.13, "resilience": 0.17, "sector": 0.15}
SECTOR_TEMPLATE_WEIGHT = 0.15

REGIME_META_TEMPLATES = {
    "CRASH":     {"event": 0.14, "flow": 0.24, "activity": 0.14, "liquidity": 0.12, "resilience": 0.36},
    "STRESS":    {"event": 0.16, "flow": 0.25, "activity": 0.16, "liquidity": 0.13, "resilience": 0.30},
    "ROTATION":  {"event": 0.16, "flow": 0.28, "activity": 0.18, "liquidity": 0.15, "resilience": 0.23},
    "EXPANSION": {"event": 0.18, "flow": 0.31, "activity": 0.23, "liquidity": 0.15, "resilience": 0.13},
    "QUIET":     {"event": 0.14, "flow": 0.24, "activity": 0.22, "liquidity": 0.20, "resilience": 0.20},
    "NORMAL":    {"event": 0.20, "flow": 0.25, "activity": 0.20, "liquidity": 0.15, "resilience": 0.20},
}
# Sektör akış ailesi tüm rejim şablonlarına eklenir (mevcut aileler orantılı küçültülür)
for _r, _w in REGIME_META_TEMPLATES.items():
    REGIME_META_TEMPLATES[_r] = {**{k: round(v * (1 - SECTOR_TEMPLATE_WEIGHT), 4) for k, v in _w.items()},
                                 "sector": SECTOR_TEMPLATE_WEIGHT}

REGIME_MIN_SCORES = {"CRASH": 88.0, "STRESS": 83.0, "ROTATION": 77.0, "EXPANSION": 74.0, "QUIET": 72.0, "NORMAL": 75.0}

FAMILY_COLS = {"event": "event_score", "flow": "flow_score", "activity": "activity_score",
               "liquidity": "liquidity_score", "resilience": "resilience_score", "sector": "sector_score"}


def profile_gate(profile):
    """'UP' = yalnızca yükselen günde aday; 'ANY' = yön kapısı yok (akış/likidite/risk kapıları sürer)."""
    return "ANY" if str((profile or {}).get("gate", "UP")).upper() == "ANY" else "UP"


def profile_signs(profile):
    s = (profile or {}).get("signs") or {}
    return {k: (-1 if _num(s.get(k), 1.0) < 0 else 1) for k in DEFAULT_META_WEIGHTS}

# Geriye uyumluluk (eski sürüm importları kırılmasın)
DEFAULT_THRESHOLDS = {"th_vol": 1.5, "th_range": 1.5, "th_flow": 2.0, "th_lambda": 1.2}
DEFAULT_WEIGHTS = {"vol": 0.35, "flow": 0.35, "range": 0.20, "lambda": 0.10}


def _num(value, default=0.0):
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def normalize_weights(weights, fallback=None):
    fallback = fallback or DEFAULT_META_WEIGHTS
    out = {k: max(_num((weights or {}).get(k), fallback[k]), 0.0) for k in fallback}
    total = sum(out.values())
    return dict(fallback) if total <= 0 else {k: v / total for k, v in out.items()}


def default_profile(regime="NORMAL"):
    regime = str(regime).upper()
    return {"weights": dict(REGIME_META_TEMPLATES.get(regime, DEFAULT_META_WEIGHTS)),
            "min_score": REGIME_MIN_SCORES.get(regime, 75.0), "regime": regime, "version": 2}


def default_profiles():
    return {r: default_profile(r) for r in C.REGIMES}


def runtime_weights(profile, regime, confidence):
    """Öğrenilmiş profil ile rejim şablonunun güvene göre karışımı (canlı + backtest ortak)."""
    template = REGIME_META_TEMPLATES.get(regime, DEFAULT_META_WEIGHTS)
    learned = normalize_weights((profile or {}).get("weights", {}), template)
    blend = 0.25 + 0.55 * float(np.clip(confidence, 0.0, 1.0))
    return normalize_weights({k: (1 - blend) * learned[k] + blend * template[k] for k in DEFAULT_META_WEIGHTS}, template)


def score_frame(df: pd.DataFrame, profiles: dict | None = None, threshold_offset: float = 0.0,
                extra_threshold_add: float = 0.0) -> pd.DataFrame:
    """Vektörel skor. df: features.build_features + attach_regime çıktısı (bir veya çok tarih)."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    profiles = profiles or {}
    out = df.copy()
    if "regime_label" not in out.columns:
        out["regime_label"] = "NORMAL"
    if "regime_confidence" not in out.columns:
        out["regime_confidence"] = 0.35
    if "macro_threshold_add" not in out.columns:
        out["macro_threshold_add"] = 0.0

    meta = pd.Series(0.0, index=out.index)
    if "eligible_any" not in out.columns:
        out["eligible_any"] = out["eligible"]
    # UP kapısı her zaman eligible_any & current_positive'tan türetilir (tekrar skorlamada kayma olmaz)
    elig = (out["eligible_any"].fillna(False).astype(bool) & out["current_positive"].fillna(False).astype(bool)).copy()
    min_score = pd.Series(75.0, index=out.index)
    wcols = {k: pd.Series(0.0, index=out.index) for k in DEFAULT_META_WEIGHTS}
    if "sector_score" not in out.columns:
        out["sector_score"] = 50.0
    for (reg, conf), idx in out.groupby(["regime_label", "regime_confidence"]).groups.items():
        prof = profiles.get(reg) or default_profile(reg)
        w = runtime_weights(prof, reg, conf)
        sg = profile_signs(prof)
        if profile_gate(prof) == "ANY":
            elig.loc[idx] = out.loc[idx, "eligible_any"].fillna(False).astype(bool)
        part = out.loc[idx]
        meta.loc[idx] = sum((part[FAMILY_COLS[k]].fillna(50.0) if sg[k] > 0 else 100.0 - part[FAMILY_COLS[k]].fillna(50.0)) * w[k]
                            for k in w)
        min_score.loc[idx] = _num(prof.get("min_score"), REGIME_MIN_SCORES.get(reg, 75.0))
        for k in w:
            wcols[k].loc[idx] = w[k]

    risk = out["overnight_risk"].fillna(100.0)
    score = meta - ((risk - 65.0) * 0.18).clip(0.0, 10.0)
    score -= np.where(out["is_downtrend_knife"].fillna(False), 15.0, 0.0)
    exc_pos = out["excess_return"].fillna(0.0) > 0.0
    score += np.where(out["regime_label"] == "CRASH", np.where(exc_pos, 5.0, -5.0), 0.0)
    score += np.where(out["regime_label"] == "STRESS", np.where(exc_pos, 3.0, -2.0), 0.0)
    score = score.clip(0.0, 99.5).round(1)

    eff = (min_score + out["macro_threshold_add"].fillna(0.0) + float(threshold_offset) + float(extra_threshold_add))
    out["meta_score"] = meta.clip(0, 99.5).round(1)
    out["risk_adjusted_score"] = score
    out["watch_score"] = score
    out["confidence_score"] = score
    out["eligible"] = elig
    out["shock_score"] = np.where(elig, score, 0.0)
    out["effective_min_score"] = eff.clip(60.0, 101.0).round(1)
    out["meta_selection"] = np.where(out["shock_score"] >= out["effective_min_score"], "SELECTED",
                                     np.where(out["watch_score"] >= out["effective_min_score"] - 5.0, "WATCH", "REJECTED"))
    out["meta_regime"] = out["regime_label"]
    out["meta_regime_confidence"] = out["regime_confidence"]
    for k in DEFAULT_META_WEIGHTS:
        out[f"meta_weight_{k}"] = wcols[k].round(4)
    out["non_price_score"] = (out["event_score"] * 0.30 + out["flow_score"] * 0.30
                              + out["activity_score"] * 0.20 + out["liquidity_score"] * 0.20).round(1)
    out["crash_resilient"] = (out["current_positive"] & exc_pos & (out["resilience_score"] >= 55.0)
                              & (out["liquidity_score"] >= 35.0))
    out["crash_survivor"] = out["crash_resilient"] & (out["resilience_score"] >= 70.0) & (out["rel_daily_pct"] >= 70.0)
    return out


def entry_status(row):
    ch = _num(row.get("change_%"))
    if 1.0 <= ch <= 4.5:
        return "🎯 UYGUN GİRİŞ BÖLGESİ"
    if ch > 6.0:
        return "🔥 UZUN VADELİ DESTEK VAR" if _num(row.get("perf_1m")) > 0 and _num(row.get("perf_3m")) > 0 else "⚠️ UZAMIŞ HAREKET"
    return "NORMAL GİRİŞ"


def stars_for(row):
    score, eff = _num(row.get("shock_score")), _num(row.get("effective_min_score"), 75.0)
    if score < eff:
        return "⭐"
    res, risk = _num(row.get("resilience_score")), _num(row.get("overnight_risk"), 100.0)
    if score >= eff + 8.0 and res >= 70.0 and risk < 62.0:
        return "⭐⭐⭐⭐⭐"
    if res >= 55.0 and risk < 72.0:
        return "⭐⭐⭐⭐"
    return "⭐⭐⭐"


def gecmis_veriyi_yukle():
    if not os.path.exists(GECMIS_DOSYA):
        return pd.DataFrame()
    try:
        df = pd.read_csv(GECMIS_DOSYA)
        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()
