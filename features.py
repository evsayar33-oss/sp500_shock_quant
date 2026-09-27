"""Özellik mühendisliği + etiketler — canlı skor ve backtest için TEK kaynak.

Tüm hesaplar (tarih x hisse) geniş matrisler üzerinde vektörel yapılır; bu yüzden
3 yıllık ~450 hisselik panel saniyeler içinde işlenir.

Önemli tasarım ilkeleri
-----------------------
* Gerçek zaman-serisi z-skorları: her hisse KENDİ geçmiş Z_WINDOW gününe göre normalize edilir.
  Taban pencere shift(1) ile bugünü içermez (ileriye bakma yok).
* Akış (flow) ailesi dürüstçe adlandırılmıştır: OHLCV tabanlı birikim vekilidir
  (Chaikin Money Flow + mum baskısı). Emir defteri verisi DEĞİLDİR.
* Overnight risk, hissenin gerçekleşmiş gece boşluğu (gap) volatilitesinden ölçülür.
* Etiket = canlı işlemle birebir: sinyal T kapanış, giriş T+1 AÇILIŞ, çıkış T+H KAPANIŞ,
  likiditeye bağlı gidiş-dönüş maliyet düşülmüş NET getiri.
* Bilanço karartması: [T, T+H] penceresinde bilanço açıklaması olan satırlar işlem dışıdır
  (canlıda planlanan, backtest'te gerçekleşmiş tarihlerle — aynı kural).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from regime import classify_market_regime, combine_regime, macro_snapshot

FAMILIES = ("event", "flow", "activity", "liquidity", "resilience")


def _wide(panel, col):
    return panel.pivot_table(index="tarih", columns="ticker", values=col, aggfunc="last").sort_index()


def _tsz(x, window=C.Z_WINDOW, min_periods=C.Z_MIN_PERIODS, lo=-4.0, hi=6.0):
    base = x.shift(1).rolling(window, min_periods=min_periods)
    mu, sd = base.mean(), base.std()
    z = (x - mu) / sd.replace(0.0, np.nan)
    return z.clip(lo, hi)


def _rank(w, valid):
    r = w.where(valid).rank(axis=1, pct=True, method="average") * 100.0
    return r.where(valid).fillna(50.0).where(valid)


def build_features(panel: pd.DataFrame, with_labels: bool = True, earnings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Panel (tarih,ticker,OHLCV) -> uzun format özellik + aile skorları (+ etiketler)."""
    if panel is None or panel.empty:
        return pd.DataFrame()

    o, h, l, c, v = (_wide(panel, k) for k in ("open", "high", "low", "close", "volume"))
    v = v.fillna(0.0)
    prev_c = c.shift(1)

    ret1 = c / prev_c - 1.0
    rng = (h - l)
    valid = c.notna() & (v > 0) & (h >= l) & (ret1.abs() <= C.MAX_VALID_DAILY_MOVE) & prev_c.notna()

    value_traded = c * v
    liq20 = value_traded.where(value_traded > 0).rolling(C.LIQ_WINDOW, min_periods=5).median()

    tr = np.fmax(np.fmax(rng, (h - prev_c).abs()), (l - prev_c).abs())
    atr_incl = tr.rolling(C.ATR_WINDOW, min_periods=5).mean()           # stop/hedef için (bugün dahil)
    atr_prior = atr_incl.shift(1)

    vol20 = ret1.rolling(20, min_periods=10).std() * 100.0
    rvol = v / v.rolling(20, min_periods=5).mean().shift(1).replace(0.0, np.nan)
    perf_1m = (c / c.shift(21) - 1.0) * 100.0
    perf_3m = (c / c.shift(63) - 1.0) * 100.0
    gap = (o / prev_c - 1.0).abs() * 100.0
    gap_vol = gap.rolling(C.GAP_WINDOW, min_periods=8).mean()

    # --- Gerçek zaman serisi z-skorları ---
    z_vol = _tsz(np.log1p(v))
    z_range = _tsz(np.log((tr / atr_prior).clip(lower=1e-3)))
    amihud = ret1.abs() * 100.0 / (value_traded / 1e6).replace(0.0, np.nan)
    z_lambda = _tsz(np.log1p(amihud))

    safe_rng = rng.replace(0.0, np.nan)
    clv = (((c - l) - (h - c)) / safe_rng).fillna(0.0)
    body = ((c - o) / safe_rng).fillna(0.0)
    candle = 0.55 * clv + 0.45 * body                                    # [-1, 1]
    z_flow = _tsz(candle)
    cmf20 = (clv * v).rolling(C.CMF_WINDOW, min_periods=8).sum() / v.rolling(C.CMF_WINDOW, min_periods=8).sum().replace(0.0, np.nan)
    close_loc = (((c - l) / safe_rng) * 100.0).fillna(50.0).clip(0, 100)

    # --- Kesitsel aile skorları (her gün kendi içinde yüzdelik) ---
    rk = lambda w: _rank(w, valid)
    pct_range, pct_lambda, pct_vol, pct_rvol = rk(z_range), rk(z_lambda), rk(z_vol), rk(rvol)
    event = pct_range * 0.35 + pct_lambda * 0.30 + pct_vol * 0.20 + pct_rvol * 0.15
    flow = rk(cmf20) * 0.60 + rk(z_flow) * 0.40
    activity = pct_rvol * 0.60 + pct_vol * 0.40
    liquidity = rk(liq20)

    chg = ret1 * 100.0
    mkt_median = chg.where(valid).median(axis=1)
    excess = chg.sub(mkt_median, axis=0)
    rel_daily, excess_pct = rk(chg), rk(excess)
    rel_1m, rel_3m = rk(perf_1m), rk(perf_3m)
    trend_persistence = rel_1m * 0.45 + rel_3m * 0.55
    resilience = (rel_daily * 0.30 + excess_pct * 0.15 + trend_persistence * 0.10 + pct_rvol * 0.10
                  + flow * 0.10 + close_loc.where(valid) * 0.10 + rel_3m * 0.15)
    overnight_risk = (rk(gap_vol) * 0.35 + rk(vol20) * 0.20 + (100.0 - liquidity) * 0.25
                      + (100.0 - flow) * 0.20).clip(0, 100)

    cols = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "change_%": chg, "value_traded": value_traded, "liq20": liq20, "rvol": rvol,
        "atr": atr_incl, "perf_1m": perf_1m, "perf_3m": perf_3m, "volatility": vol20,
        "gap_vol": gap_vol, "z_vol": z_vol, "z_range": z_range, "z_lambda": z_lambda,
        "z_flow": z_flow, "candle_pressure": candle, "cmf20": cmf20, "close_location": close_loc,
        "excess_return": excess, "rel_daily_pct": rel_daily, "rel_1m_pct": rel_1m, "rel_3m_pct": rel_3m,
        "rel_rvol_pct": pct_rvol, "trend_persistence": trend_persistence,
        "event_score": event, "flow_score": flow, "activity_score": activity,
        "liquidity_score": liquidity, "resilience_score": resilience, "overnight_risk": overnight_risk,
        "valid": valid,
    }

    if with_labels:
        H = C.HORIZON
        entry = o.shift(-1)
        exit_ = c.shift(-H)
        inv = (~valid).astype(float)
        fut_bad = sum(inv.shift(-k).fillna(1.0) for k in range(1, H + 1))
        label_ok = (fut_bad == 0) & entry.notna() & exit_.notna() & (entry > 0)
        gross = (exit_ / entry - 1.0) * 100.0
        cost = pd.DataFrame(C.round_trip_cost_pct(liq20.fillna(C.MIN_LIQ_TL).values), index=liq20.index, columns=liq20.columns)
        cols["entry_next_open"] = entry.where(label_ok)
        cols["gross_ret"] = gross.where(label_ok)
        cols["cost_rt"] = cost
        cols["net_ret"] = (gross - cost).where(label_ok)
        cols["gap_next"] = ((entry / c - 1.0) * 100.0).where(label_ok)

    stacked = {k: w.stack() if hasattr(w, "stack") else w for k, w in cols.items()}
    df = pd.DataFrame(stacked)
    df.index.names = ["tarih", "ticker"]
    df = df.reset_index()
    df = df[df["valid"].fillna(False).astype(bool)].copy()

    df["market_median_change"] = df["tarih"].map(mkt_median)
    df["is_downtrend_knife"] = (df["perf_3m"] < -25.0) & (df["perf_1m"] < -10.0)
    df["is_illiquid"] = df["liq20"].fillna(0.0) < C.MIN_LIQ_TL
    df["current_positive"] = df["change_%"] > 0.0
    df["directional_flow_ok"] = df["flow_score"] >= C.MIN_FLOW_SCORE
    df["earnings_in_window"] = mark_earnings_window(df, earnings)
    df["eligible"] = (df["current_positive"] & df["directional_flow_ok"] & ~df["is_downtrend_knife"]
                      & ~df["is_illiquid"] & (df["overnight_risk"] < C.MAX_OVERNIGHT_RISK))
    if C.EARNINGS_BLACKOUT:
        df["eligible"] &= ~df["earnings_in_window"]
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


def mark_earnings_window(df: pd.DataFrame, earnings: pd.DataFrame | None) -> pd.Series:
    """Her (tarih, hisse) için [T, T+HORIZON iş günü] içinde bilanço var mı?"""
    out = pd.Series(False, index=df.index)
    if earnings is None or earnings.empty or df.empty:
        return out
    ed = earnings.dropna()
    ed = {t: np.sort(g["date"].values.astype("datetime64[D]")) for t, g in ed.groupby("ticker")}
    for t, idx in df.groupby("ticker").groups.items():
        dates = ed.get(t)
        if dates is None or len(dates) == 0:
            continue
        d = df.loc[idx, "tarih"].values.astype("datetime64[D]")
        end = np.busday_offset(d, C.HORIZON, roll="forward")
        pos = np.searchsorted(dates, d, side="left")
        nxt = np.where(pos < len(dates), dates[np.minimum(pos, len(dates) - 1)], np.datetime64("NaT"))
        out.loc[idx] = (pos < len(dates)) & (nxt <= end)
    return out


def regime_by_date(feat: pd.DataFrame, macro_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Her tarih için nihai rejim (kesitsel + makro). Yalnızca o günün verisini kullanır."""
    rows = []
    if feat is None or feat.empty:
        return pd.DataFrame()
    liquid = feat[~feat["is_illiquid"]]
    for day, grp in liquid.groupby("tarih", sort=True):
        cross = classify_market_regime(grp)
        snap = combine_regime(cross, macro_snapshot(macro_frame, day))
        rows.append({
            "tarih": day, "regime_label": snap["label"], "regime_confidence": snap["confidence"],
            "cross_label": snap["cross_label"], "macro_label": snap["macro_label"],
            "macro_stress": snap["macro_stress"], "macro_threshold_add": snap["macro_threshold_add"],
            "macro_exposure_mult": snap["macro_exposure_mult"],
        })
    return pd.DataFrame(rows)


def attach_regime(feat: pd.DataFrame, macro_frame=None) -> pd.DataFrame:
    reg = regime_by_date(feat, macro_frame)
    if reg.empty:
        feat = feat.copy()
        feat["regime_label"], feat["regime_confidence"] = "NORMAL", 0.35
        feat["macro_label"], feat["macro_stress"], feat["macro_threshold_add"] = "UNKNOWN", 0.0, 0.0
        feat["macro_exposure_mult"] = 1.0
        return feat
    out = feat.merge(reg, on="tarih", how="left")
    fills = {"regime_label": "NORMAL", "regime_confidence": 0.35, "cross_label": "NORMAL", "macro_label": "UNKNOWN",
             "macro_stress": 0.0, "macro_threshold_add": 0.0, "macro_exposure_mult": 1.0}
    return out.fillna(value=fills)


def correlation_matrix(panel: pd.DataFrame, tickers, end_date=None, window=C.CORR_WINDOW):
    """Seçilen hisseler için son `window` günlük getiri korelasyonu."""
    if panel is None or panel.empty or not tickers:
        return pd.DataFrame()
    sub = panel[panel["ticker"].isin(tickers)]
    if end_date is not None:
        sub = sub[sub["tarih"] <= pd.Timestamp(end_date)]
    c = sub.pivot_table(index="tarih", columns="ticker", values="close").sort_index().tail(window + 1)
    return c.pct_change().corr(min_periods=20)
