"""Meta-etiketleme (meta-labeling) — "bu sinyal kazanır mı?" olasılık filtresi.

Kurumsal kantitatif fonların win-rate'i yükseltmek için kullandığı ikinci katman
(López de Prado, 2018). Birinci katman (skor) hangi hisselerin ADAY olduğunu söyler;
ikinci katman her aday için P(kazanç) tahmin eder ve yalnızca olasılığı yüksek olanları geçirir.

* Model: L2-cezalı lojistik regresyon (saf numpy, ek kütüphane yok, aşırı öğrenmeye dayanıklı).
* Girdi: aile skorları, grup akışı, mum/akış yapısı, volatilite, rejim ve makro durumu.
* Hedef: net_ret > 0 (T+1 açılış -> T+HORIZON kapanış, maliyet düşülmüş) — canlıyla aynı tanım.
* Eşik p* ve skor bandı YALNIZCA eğitim verisinde seçilir (Wilson LCB maksimizasyonu).
* Devreye alma: walk-forward OOS'ta win-rate LCB'yi yükseltip net getiriyi bozmuyorsa
  sistem kendiliğinden açar; aksi halde kapalı kalır (gerçek veride S&P'de açıldı, BIST'te kapalı).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

FEATS = ["event_score", "flow_score", "activity_score", "liquidity_score", "resilience_score", "sector_score",
         "watch_score", "sec_cmf", "sec_ret20", "sec_acc", "own_rel5", "lead_gap", "z_vol", "z_range", "cmf20",
         "candle_pressure", "close_location", "change_%", "volatility", "gap_vol", "rel_1m_pct", "rel_3m_pct",
         "macro_stress", "regime_confidence", "excess_return"]
L2 = 3.0
BANDS = (0.0, 6.0)                      # seçimde: skor >= eşik - bant (bant eğitimde seçilir)
TRAIN_BAND = 1000.0                     # eğitim havuzu: TÜM uygun hisseler. Gerçek veride (S&P, OOS) dar havuz
                                        # (bant 6) WR %45.8 verirken tüm havuz WR %56.8 verdi: model kaybedenleri
                                        # tanımak için eşik altındaki örneklere de ihtiyaç duyuyor.
Q_GRID = [round(x, 2) for x in np.arange(0.45, 0.68, 0.01)]
MIN_TRAIN_TRADES = 150


def _design(df: pd.DataFrame, mu=None, sd=None):
    X = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce") if f in df.columns else 0.0 for f in FEATS},
                     index=df.index).astype(float)
    if mu is None:
        mu, sd = X.mean(), X.std().replace(0.0, 1.0).fillna(1.0)
        mu = mu.fillna(0.0)
    Z = ((X - mu) / sd).clip(-4, 4).fillna(0.0)
    reg = pd.get_dummies(df.get("regime_label", pd.Series("NORMAL", index=df.index))).reindex(
        columns=list(C.REGIMES), fill_value=0).astype(float)
    return np.hstack([Z.values, reg.values, np.ones((len(df), 1))]), mu, sd


def _fit(X, y, l2=L2, iters=60):
    w = np.zeros(X.shape[1])
    n = max(len(y), 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / n + l2 * w / n
        h = (X * (p * (1 - p))[:, None]).T @ X / n + np.eye(len(w)) * l2 / n
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def predict(model: dict, df: pd.DataFrame) -> pd.Series:
    if not model or df is None or df.empty:
        return pd.Series(np.nan, index=getattr(df, "index", None))
    X, _, _ = _design(df, pd.Series(model["mu"]), pd.Series(model["sd"]))
    w = np.array(model["w"])
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30))), index=df.index)


def candidate_pool(scored: pd.DataFrame, band: float = max(BANDS)) -> pd.DataFrame:
    """Meta modelin gördüğü aday havuzu: uygun + skor >= eşik - bant."""
    if scored is None or scored.empty:
        return pd.DataFrame()
    return scored[scored["eligible"] & (scored["watch_score"] >= scored["effective_min_score"] - band)]


def select_with_meta(scored: pd.DataFrame, model: dict, top_k=C.TOP_K_PER_DAY) -> pd.DataFrame:
    pool = candidate_pool(scored, model.get("band", 0.0)).copy()
    if pool.empty:
        return pool
    pool["p_win"] = predict(model, pool)
    sel = pool[pool["p_win"] >= model["q"]]
    return sel.sort_values(["tarih", "p_win"], ascending=[True, False]).groupby("tarih", sort=False).head(top_k)


def fit_meta(train_scored: pd.DataFrame, metrics_fn) -> dict | None:
    """Eğitim havuzunda modeli kurar, p* ve bandı eğitimde seçer. Yetersiz veri -> None."""
    pool = candidate_pool(train_scored, TRAIN_BAND)
    pool = pool[pool["net_ret"].notna()]
    if len(pool) < MIN_TRAIN_TRADES * 2 or pool["tarih"].nunique() < 60:
        return None
    X, mu, sd = _design(pool)
    y = (pool["net_ret"] > 0).astype(float).values
    w = _fit(X, y)
    pool = pool.assign(p_win=1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30))))
    best = None
    for band in BANDS:
        sub = pool[pool["watch_score"] >= pool["effective_min_score"] - band]
        if len(sub) < MIN_TRAIN_TRADES:
            continue
        for q in Q_GRID:
            s = sub[sub["p_win"] >= q]
            if len(s) < MIN_TRAIN_TRADES:
                break
            s = s.sort_values(["tarih", "p_win"], ascending=[True, False]).groupby("tarih").head(C.TOP_K_PER_DAY)
            m = metrics_fn(s)
            if m["n"] < MIN_TRAIN_TRADES or m["avg_return"] <= 0:
                continue
            key = (m["wilson_lcb"], m["avg_return"])
            if best is None or key > best[0]:
                best = (key, band, q, m)
    if best is None:
        return None
    return {"w": [float(v) for v in w], "mu": {k: float(v) for k, v in mu.items()},
            "sd": {k: float(v) for k, v in sd.items()}, "band": float(best[1]), "q": float(best[2]),
            "train_metrics": best[3], "feats": FEATS, "n_train": int(len(pool))}
