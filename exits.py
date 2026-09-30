"""Çıkış motoru v3 — backtest etiketi ve canlı pozisyon takibi için TEK kural.

Kural (seviyeler giriş fiyatı ve sinyal günü ATR'sine göre):
  STOP = giriş − EXIT_STOP_ATR × ATR
  TP1  = giriş + EXIT_TP1_ATR  × ATR  -> pozisyonun EXIT_TP1_FRAC'ı satılır, stop GİRİŞE çekilir (başabaş)
  TP2  = giriş + EXIT_TP2_ATR  × ATR  -> kalan kısım satılır
  Hiçbiri olmazsa kalan kısım HORIZON. günün kapanışında satılır.

Günlük bar sırası (temkinli):
  * Gün 2+ açılışı: stopun altında açılırsa açılıştan stop; hedefin üstünde açılırsa açılıştan hedef.
  * Gün içi: TP1 öncesinde aynı gün hem stop hem TP1'e değdiyse ÖNCE STOP varsayılır.
  * TP1 ile aynı gün düşük fiyat girişe indiyse (ve TP2 gelmediyse) kalan kısım başabaş kapanır.
Bu kural gerçek veride (BIST, S&P) işlem bazında simülatörle birebir doğrulandı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

REASONS = {0: "SÜRE", 1: "STOP", 2: "BAŞABAŞ", 3: "TP1", 4: "TP2"}


def params():
    return {"s": float(getattr(C, "EXIT_STOP_ATR", 2.0)), "a": float(getattr(C, "EXIT_TP1_ATR", 1.5)),
            "b": getattr(C, "EXIT_TP2_ATR", 2.0), "f": float(getattr(C, "EXIT_TP1_FRAC", 0.5))}


def levels(entry: float, atr_pct: float, p: dict | None = None) -> dict:
    p = p or params()
    A = atr_pct * entry
    return {"stop": entry - p["s"] * A, "tp1": entry + p["a"] * A,
            "tp2": (entry + float(p["b"]) * A) if p["b"] else None}


# ------------------------------------------------------------------
# Tek işlem (canlı defter) — kısmi yol da işlenir
# ------------------------------------------------------------------
def run_path(O, Hh, L, Cc, entry: float, atr_pct: float, horizon: int = None, p: dict | None = None) -> dict:
    """O,H,L,C: girişten itibaren elde olan günler (uzunluk <= HORIZON).
    Dönüş: closed, net_gross(%), tp1, tp2, stop, reason, exit_day, rem, stop_level, mtm(%)."""
    p = p or params()
    horizon = horizon or C.HORIZON
    lv = levels(entry, atr_pct, p)
    e, stop, t1 = entry, lv["stop"], lv["tp1"]
    t2 = lv["tp2"] if lv["tp2"] is not None else np.inf
    q1 = p["f"] if p["b"] else 1.0
    rem, pnl, hit1, tp2, st, reason, xday = 1.0, 0.0, False, False, False, None, None
    t1day = None
    n = len(Cc)
    for j in range(n):
        o_, h_, l_, c_ = float(O[j]), float(Hh[j]), float(L[j]), float(Cc[j])
        if j > 0:
            if o_ <= stop:
                pnl += rem * (o_ / e - 1); rem = 0; st = not hit1; reason = 1 if not hit1 else 2; xday = j; break
            if not hit1 and o_ >= t1:
                pnl += q1 * (o_ / e - 1); rem -= q1; hit1 = True; stop = max(stop, e); t1day = j
                if rem <= 1e-9:
                    reason, xday = 3, j; break
            if hit1 and o_ >= t2:
                pnl += rem * (o_ / e - 1); rem = 0; tp2 = True; reason, xday = 4, j; break
        if not hit1 and l_ <= stop:
            pnl += rem * (stop / e - 1); rem = 0; st = True; reason, xday = 1, j; break
        if not hit1 and h_ >= t1:
            pnl += q1 * (t1 / e - 1); rem -= q1; hit1 = True; stop = max(stop, e); t1day = j
            if rem <= 1e-9:
                reason, xday = 3, j; break
            if h_ >= t2:
                pnl += rem * (t2 / e - 1); rem = 0; tp2 = True; reason, xday = 4, j; break
            if l_ <= stop:
                rem = 0; reason, xday = 2, j; break
            continue
        if hit1:
            if l_ <= stop:
                pnl += rem * (stop / e - 1); rem = 0; reason, xday = 2, j; break
            if h_ >= t2:
                pnl += rem * (t2 / e - 1); rem = 0; tp2 = True; reason, xday = 4, j; break
    closed = rem <= 1e-9
    if not closed and n >= horizon:
        pnl += rem * (float(Cc[horizon - 1]) / e - 1); rem = 0; closed = True; reason, xday = 0, horizon - 1
    mtm = pnl + (rem * (float(Cc[n - 1]) / e - 1) if (n and not closed) else 0.0)
    return {"closed": closed, "gross": pnl * 100.0, "mtm": mtm * 100.0, "tp1": hit1, "tp2": tp2, "stop": st,
            "reason": REASONS.get(reason) if closed else ("TP1 ✓ (stop girişte)" if hit1 else "AÇIK"),
            "exit_day": (xday + 1) if xday is not None else None, "rem": rem, "stop_level": stop,
            "tp1_level": lv["tp1"], "tp2_level": lv["tp2"], "days": n,
            "tp1_day": (t1day + 1) if t1day is not None else None}


# ------------------------------------------------------------------
# Vektörel etiket (backtest) — geniş matrisler üzerinde
# ------------------------------------------------------------------
def wide_labels(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, atr_pct: pd.DataFrame,
                p: dict | None = None, horizon: int = None) -> dict:
    """Her (T, hisse) sinyali için: giriş T+1 açılış, çıkış kuralı ile brüt getiri (%), bayraklar, çıkış nedeni."""
    p = p or params()
    Hn = horizon or C.HORIZON
    idx, cols = c.index, c.columns
    E = o.shift(-1).values
    A = atr_pct.values * E
    stop = E - p["s"] * A
    t1 = E + p["a"] * A
    t2 = (E + float(p["b"]) * A) if p["b"] else np.full_like(E, np.inf)
    q1 = p["f"] if p["b"] else 1.0
    shape = E.shape
    rem = np.ones(shape); pnl = np.zeros(shape)
    hit1 = np.zeros(shape, bool); tp2f = np.zeros(shape, bool); stf = np.zeros(shape, bool)
    reason = np.full(shape, -1); xday = np.full(shape, -1)
    O = [o.shift(-(j + 1)).values for j in range(Hn)]
    Hh = [h.shift(-(j + 1)).values for j in range(Hn)]
    Lw = [l.shift(-(j + 1)).values for j in range(Hn)]
    Cw = [c.shift(-(j + 1)).values for j in range(Hn)]
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(Hn):
            oj, hj, lj = O[j], Hh[j], Lw[j]
            act = rem > 1e-9
            if j > 0:
                m = act & (oj <= stop)
                pnl[m] += rem[m] * (oj[m] / E[m] - 1); stf[m] = ~hit1[m]; reason[m] = np.where(hit1[m], 2, 1); xday[m] = j; rem[m] = 0
                act = rem > 1e-9
                m = act & ~hit1 & (oj >= t1)
                pnl[m] += q1 * (oj[m] / E[m] - 1); rem[m] -= q1; hit1[m] = True; stop[m] = np.maximum(stop[m], E[m])
                z = m & (rem <= 1e-9); reason[z] = 3; xday[z] = j
                act = rem > 1e-9
                m = act & hit1 & (oj >= t2)
                pnl[m] += rem[m] * (oj[m] / E[m] - 1); tp2f[m] = True; reason[m] = 4; xday[m] = j; rem[m] = 0
                act = rem > 1e-9
            m = act & ~hit1 & (lj <= stop)
            pnl[m] += rem[m] * (stop[m] / E[m] - 1); stf[m] = True; reason[m] = 1; xday[m] = j; rem[m] = 0
            act = rem > 1e-9
            b = act & ~hit1 & (hj >= t1)
            pnl[b] += q1 * (t1[b] / E[b] - 1); rem[b] -= q1; hit1[b] = True; stop[b] = np.maximum(stop[b], E[b])
            z = b & (rem <= 1e-9); reason[z] = 3; xday[z] = j
            b = b & (rem > 1e-9)
            m = b & (hj >= t2)
            pnl[m] += rem[m] * (t2[m] / E[m] - 1); tp2f[m] = True; reason[m] = 4; xday[m] = j; rem[m] = 0
            m = b & (rem > 1e-9) & (lj <= stop)
            reason[m] = 2; xday[m] = j; rem[m] = 0
            act = (rem > 1e-9) & hit1 & ~b
            m = act & (lj <= stop)
            pnl[m] += rem[m] * (stop[m] / E[m] - 1); reason[m] = 2; xday[m] = j; rem[m] = 0
            m = act & ~(lj <= stop) & (hj >= t2)
            pnl[m] += rem[m] * (t2[m] / E[m] - 1); tp2f[m] = True; reason[m] = 4; xday[m] = j; rem[m] = 0
        cl = Cw[Hn - 1]
        m = rem > 1e-9
        pnl[m] += rem[m] * (cl[m] / E[m] - 1); reason[m] = 0; xday[m] = Hn - 1
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return {"gross": mk(pnl * 100.0), "tp1": mk(hit1), "tp2": mk(tp2f), "stop": mk(stf),
            "reason": mk(reason), "exit_day": mk(xday + 1)}
