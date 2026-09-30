"""Portföy inşası: volatilite hedefli pozisyon boyutu, korelasyon filtresi, brüt limit, ATR çıkış seviyeleri.

Pozisyon ağırlığı:  w = RISK_PER_TRADE / σ_H   (σ_H = günlük vol * sqrt(HORIZON))
  -> her pozisyon HORIZON günde yaklaşık aynı 1σ risk (özsermayenin %RISK_PER_TRADE'i) taşır.
Ardından: kanaat (skor marjı) çarpanı, rejim/makro/otonomi maruziyet çarpanları,
tekil üst/alt sınır, istatistiksel grup (küme) başına MAX_PER_GROUP, resmi sektör başına MAX_PER_SECTOR
(varsa), MAX_POSITIONS ve brüt MAX_GROSS limiti uygulanır. Sektör akışı adayları aynı gruba
yığabileceği için grup limiti konsantrasyon riskine karşı zorunludur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


MAX_PER_SECTOR = getattr(C, "MAX_PER_SECTOR", 3)
MAX_PER_GROUP = getattr(C, "MAX_PER_GROUP", 3)


def _has(x):
    try:
        return x is not None and not (isinstance(x, float) and np.isnan(x)) and str(x) != ""
    except Exception:
        return False


def _num(x, d=0.0):
    try:
        x = float(x)
        return d if not np.isfinite(x) else x
    except Exception:
        return d


def build_portfolio(candidates: pd.DataFrame, corr: pd.DataFrame | None = None,
                    exposure_mult: float = 1.0) -> pd.DataFrame:
    """candidates: skorlanmış, eşiği geçmiş adaylar (skora göre sıralı).
    Dönen tabloda weight_pct, stop/tp seviyeleri ve (varsa) elenme nedeni bulunur."""
    if candidates is None or candidates.empty:
        return pd.DataFrame() if candidates is None else candidates.assign(weight_pct=0.0)
    df = candidates.copy().reset_index(drop=True)
    exposure_mult = float(np.clip(exposure_mult, 0.0, 1.0))
    gross_cap = C.MAX_GROSS_PCT * exposure_mult

    chosen, chosen_sec, chosen_grp, weights, reasons = [], [], [], [], []
    gross = 0.0
    for _, r in df.iterrows():
        t = r["ticker"]
        reason = ""
        if len(chosen) >= C.MAX_POSITIONS:
            reason = "MAX_POZİSYON"
        elif _has(r.get("grp")) and sum(1 for g in chosen_grp if g == r.get("grp")) >= MAX_PER_GROUP:
            reason = f"GRUP_LİMİTİ(G{int(r.get('grp'))})"
        elif _has(r.get("sector")) and sum(1 for s in chosen_sec if s == r.get("sector")) >= MAX_PER_SECTOR:
            reason = f"SEKTÖR_LİMİTİ({r.get('sector')})"
        elif corr is not None and not corr.empty and t in corr.index:
            hi = [s for s in chosen if s in corr.columns and _num(corr.at[t, s]) > C.MAX_PAIR_CORR]
            if hi:
                reason = f"KORELASYON({hi[0]} ρ>{C.MAX_PAIR_CORR:.2f})"
        w = 0.0
        if not reason:
            sigma_h = max(_num(r.get("volatility"), 3.0), 0.5) * np.sqrt(C.HORIZON)
            base = 100.0 * C.RISK_PER_TRADE_PCT / sigma_h
            margin = _num(r.get("shock_score")) - _num(r.get("effective_min_score"), 75.0)
            conviction = 0.6 + 0.4 * float(np.clip(margin / 15.0, 0.0, 1.0))
            w = float(np.clip(base * conviction, C.MIN_POSITION_PCT, C.MAX_POSITION_PCT)) * exposure_mult
            if gross + w > gross_cap:
                w = max(gross_cap - gross, 0.0)
            if w < C.MIN_POSITION_PCT * max(exposure_mult, 0.35):
                reason, w = "BRÜT_LİMİT", 0.0
        if w > 0:
            chosen.append(t)
            chosen_sec.append(r.get("sector"))
            chosen_grp.append(r.get("grp"))
            gross += w
        weights.append(round(w, 1))
        reasons.append(reason)

    df["weight_pct"] = weights
    df["skip_reason"] = reasons
    atr = pd.to_numeric(df.get("atr"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    df["stop_price"] = (close - C.STOP_ATR * atr).round(2)
    df["tp1_price"] = (close + C.TP1_ATR * atr).round(2)
    df["allocation"] = np.where(df["weight_pct"] > 0,
                                df["weight_pct"].map(lambda x: f"%{x:.1f} (vol-hedefli)"),
                                "İşlem Açma (" + df["skip_reason"].replace("", "Limit") + ")")
    return df
