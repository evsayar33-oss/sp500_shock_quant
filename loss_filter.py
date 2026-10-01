"""Kayıp filtresi (v3.3) — sistemin GERÇEK sinyallerinden öğrenilen ikinci savunma hattı.

Gerçek veri araştırması (1 Eki 2026, sistemin kendi örneklem dışı sinyalleri, ileriye dönük test):
  * Volatilite tabanı: TP1 = 1×ATR olduğu için sakin hissede kazanç küçük, maliyet sabit -> net erir.
    Sinyallerin en sakin %33'ü elendiğinde BIST net/işlem +0.78 -> +1.10, S&P +0.47 -> +0.57.
  * Kayıp modeli (yalnız BIST): "bu sinyal zararla kapanır mı?" olasılığı (güçlü L2'li lojistik).
    En riskli üçte bir elendiğinde BIST WR %68.6 -> %71.4. Plasebo (karıştırılmış özellik) hep kötüleştirdi.
  * İkisi birlikte (BIST): WR %73.2, kayıp oranı %31.4 -> %26.8, net +0.78 -> +1.45, PF 1.30 -> 1.53.

Güvenlik: filtre her denetimde iç içe (nested) walk-forward ile yeniden sınanır; açmak/kapatmak histerezisli.
Kanıt kaybolursa kendiliğinden kapanır. Saf numpy — ek kütüphane yok.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

FEATS = ["volatility", "gap_vol", "overnight_risk", "perf_1m", "perf_3m", "rel_1m_pct", "rel_3m_pct",
         "trend_persistence", "rvol", "rel_rvol_pct", "change_%", "excess_return", "rel_daily_pct", "z_vol",
         "z_range", "z_flow", "z_lambda", "cmf20", "close_location", "candle_pressure", "event_score", "flow_score",
         "activity_score", "liquidity_score", "resilience_score", "sector_score", "sec_cmf", "sec_acc", "sec_ret5",
         "sec_ret20", "own_rel5", "lead_gap", "grp_size", "macro_stress", "regime_confidence", "score_margin",
         "log_liq"]

VOL_Q = float(getattr(C, "LOSS_VOL_Q", 0.33))          # en sakin bu oran elenir
USE_MODEL = bool(getattr(C, "LOSS_MODEL", True))       # kayıp modeli (BIST açık, S&P kapalı)
DROP_Q = float(getattr(C, "LOSS_DROP_Q", 0.33))        # modelce en riskli bu oran elenir
L2 = float(getattr(C, "LOSS_L2", 50.0))                # güçlü düzenlileştirme (araştırmadaki C=0.02 karşılığı)
MIN_TRAIN = 300
MIN_LIFT = float(getattr(C, "LOSS_MIN_LIFT", 0.10))  # açmak için net/işlem iyileşmesi (puan)
MIN_FOLD_TRADES = 10
EMBARGO_DAYS = int(getattr(C, "HORIZON", 10)) + 6      # takvim günü: eğitim işlemleri test başından önce kapanmış olmalı


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "score_margin" not in d.columns:
        d["score_margin"] = pd.to_numeric(d.get("shock_score"), errors="coerce") - \
                            pd.to_numeric(d.get("effective_min_score"), errors="coerce")
    if "log_liq" not in d.columns:
        d["log_liq"] = np.log(pd.to_numeric(d.get("liq20", np.nan), errors="coerce").clip(lower=1.0))
    return d


def _design(df: pd.DataFrame, mu=None, sd=None, feats=None):
    feats = feats or FEATS
    X = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan for f in feats},
                     index=df.index).astype(float)
    if mu is None:
        mu = X.median().fillna(0.0)
        sd = X.std().replace(0.0, 1.0).fillna(1.0)
    Z = ((X.fillna(mu) - mu) / sd).clip(-4, 4).fillna(0.0)
    reg = pd.get_dummies(df.get("regime_label", pd.Series("NORMAL", index=df.index))).reindex(
        columns=list(C.REGIMES), fill_value=0).astype(float)
    return np.hstack([Z.values, reg.values, np.ones((len(df), 1))]), mu, sd


def _fit(X, y, l2=L2, iters=60):
    w = np.zeros(X.shape[1])
    n = max(len(y), 1)
    pen = np.ones(len(w)); pen[-1] = 0.0                # sabit terim cezasız
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = (X.T @ (p - y) + l2 * pen * w) / n
        h = ((X * (p * (1 - p))[:, None]).T @ X + np.diag(l2 * pen + 1e-9)) / n
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def _p_loss(model: dict, df: pd.DataFrame) -> np.ndarray:
    X, _, _ = _design(df, pd.Series(model["mu"]), pd.Series(model["sd"]), model.get("feats"))
    return 1.0 / (1.0 + np.exp(-np.clip(X @ np.array(model["w"]), -30, 30)))


def fit(trades: pd.DataFrame) -> dict | None:
    """Tüm (örneklem dışı) sistem işlemleriyle canlı filtreyi kurar."""
    if trades is None or trades.empty:
        return None
    d = _prep(trades[pd.to_numeric(trades["net_ret"], errors="coerce").notna()])
    if len(d) < MIN_TRAIN:
        return None
    vol = pd.to_numeric(d["volatility"], errors="coerce")
    model = {"vol_floor": float(np.nanquantile(vol, VOL_Q)), "vol_q": VOL_Q, "use_model": USE_MODEL,
             "n_train": int(len(d))}
    if USE_MODEL:
        X, mu, sd = _design(d)
        y = (pd.to_numeric(d["net_ret"]) < 0).astype(float).values
        w = _fit(X, y)
        model.update({"w": [float(v) for v in w], "mu": {k: float(v) for k, v in mu.items()},
                      "sd": {k: float(v) for k, v in sd.items()}, "feats": FEATS, "drop_q": DROP_Q})
        p = _p_loss(model, d)
        model["p_max"] = float(np.quantile(p, 1.0 - DROP_Q))   # bundan riskli olan elenir
    return model


def apply(df: pd.DataFrame, model: dict | None) -> tuple[pd.Series, pd.Series]:
    """(geçenler maskesi, eleme nedeni). Model yoksa hepsi geçer."""
    keep = pd.Series(True, index=df.index)
    why = pd.Series("", index=df.index, dtype=object)
    if not model or df is None or df.empty:
        return keep, why
    d = _prep(df)
    vol = pd.to_numeric(d["volatility"], errors="coerce")
    low = vol < float(model["vol_floor"])
    keep &= ~low
    why[low] = "düşük volatilite (TP1 maliyeti karşılamaz)"
    if model.get("use_model") and model.get("w"):
        p = pd.Series(_p_loss(model, d), index=df.index)
        risky = p > float(model["p_max"])
        why[risky & keep] = [f"yüksek kayıp riski (%{100 * v:.0f})" for v in p[risky & keep]]
        keep &= ~risky
    return keep, why


def walk_forward(trades: pd.DataFrame, metrics_fn) -> dict:
    """İç içe walk-forward: her dilim yalnızca ÖNCEKİ dilimlerin kapanmış işlemleriyle öğrenir."""
    if trades is None or trades.empty or "fold" not in trades.columns:
        return {"ok": False, "reason": "işlem yok"}
    d = _prep(trades[pd.to_numeric(trades["net_ret"], errors="coerce").notna()]).copy()
    d["tarih"] = pd.to_datetime(d["tarih"])
    keep = pd.Series(False, index=d.index)
    evald = pd.Series(False, index=d.index)
    folds = []
    for k in sorted(d["fold"].unique()):
        te = d["fold"] == k
        t0 = d.loc[te, "tarih"].min()
        tr = d["tarih"] < t0 - pd.Timedelta(days=EMBARGO_DAYS)
        if tr.sum() < MIN_TRAIN or te.sum() < MIN_FOLD_TRADES:
            continue
        m = fit(d[tr])
        if not m:
            continue
        kk, _ = apply(d[te], m)
        keep[te] = kk.values
        evald[te] = True
        folds.append({"fold": int(k), "base": metrics_fn(d[te]), "filt": metrics_fn(d[te & keep])})
    if not folds:
        return {"ok": False, "reason": "yetersiz geçmiş işlem"}
    return {"ok": True, "base": metrics_fn(d[evald]), "filt": metrics_fn(d[evald & keep]), "folds": folds,
            "kept_share": float(keep[evald].mean()), "trades": d[evald & keep]}


def decision(base_m: dict, filt_m: dict, folds: list, was_on: bool = False) -> tuple[bool, str]:
    """Histerezisli aç/kapa. Açmak: dilimlerin >= %60'ında net üstünlük, net >= LOSS_MIN_LIFT puan, PF >= baz, n >= 100.
    Açıkken kapatmak: dilim üstünlüğü < %50, net farkı < 0 ya da PF < baz."""
    if filt_m.get("n", 0) < 100:
        return False, f"filtre OOS örneklem yetersiz (n={filt_m.get('n', 0)})"
    fl = [f for f in folds if f["filt"].get("n", 0) >= 5 and f["base"].get("n", 0) >= MIN_FOLD_TRADES]
    wins = sum(1 for f in fl if f["filt"]["avg_return"] > f["base"]["avg_return"])
    share = wins / len(fl) if fl else 0.0
    avg = filt_m["avg_return"] - base_m["avg_return"]
    wr = filt_m["win_rate"] - base_m["win_rate"]
    pf_ok = filt_m["profit_factor"] >= base_m["profit_factor"]
    if was_on:
        ok = share >= 0.5 and avg >= 0 and pf_ok
        mode = "korunuyor" if ok else "kapatıldı"
    else:
        ok = share >= 0.6 and avg >= MIN_LIFT and pf_ok
        mode = "açıldı" if ok else "kapalı"
    return bool(ok), (f"volatilite filtresi {mode} | kazanma {wr:+.1f}pp | net {avg:+.2f} | PF {filt_m['profit_factor']:.2f} "
                      f"vs {base_m['profit_factor']:.2f} | dilim {wins}/{len(fl)}")
