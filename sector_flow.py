"""Sektör Akış Katmanı (Sector Flow Engine) — kurumsal para akışının grup düzeyindeki ayak izleri.

Neden resmi sektör etiketi değil de istatistiksel küme?
------------------------------------------------------
Kurumsal para etiketleri değil, birlikte hareket eden sepetleri (tema, faktör, endeks ağırlığı)
takip eder. Gerçek veride yapılan testte (S&P 500, 2023-2026) aynı birikim sinyali
korelasyon kümeleriyle IC +0.043 (t≈2.0), resmi sektör etiketleriyle IC +0.007 (t≈0.4) verdi.
BIST'te sektör birikimi sistemin en güçlü pozitif sinyali çıktı (IC +0.064, t≈5.0).

Kümeleme (ileriye bakmasız / point-in-time)
-------------------------------------------
* Her ayın ilk işlem gününde, önceki CLUSTER_LOOKBACK günün PİYASADAN ARINDIRILMIŞ getirileri
  (hisse getirisi - günlük kesitsel medyan) ile korelasyon matrisi kurulur.
* Uzaklık = 1 - ρ, ortalama bağlantılı hiyerarşik kümeleme, en fazla CLUSTER_K küme.
* O ay boyunca atamalar sabittir. Takvime sabitlendiği için canlı ve backtest AYNI kümeleri üretir.

Grup özellikleri (her gün, grup içinde)
---------------------------------------
sec_cmf    : grup üyelerinin ortalama Chaikin Money Flow'u   -> grup düzeyinde birikim
sec_acc    : (CMF>0 ve hacim z>0.5) üye oranı                -> birikim genişliği
sec_ret20  : grup medyan 20g getirisi - piyasa medyanı       -> grup göreli gücü
lead_gap   : grubun en likit 3 üyesinin 5g getirisi - hissenin 5g getirisi -> takipçi yetişme
own_rel5   : hissenin 5g getirisi - grup medyanı             -> tek başına sapma (geri dönme eğilimli)
sector_score = 0.45·rk(sec_cmf) + 0.20·rk(sec_ret20) + 0.15·rk(sec_acc) + 0.10·rk(lead_gap) + 0.10·rk(-own_rel5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

CLUSTER_LOOKBACK = getattr(C, "CLUSTER_LOOKBACK", 250)
CLUSTER_K = getattr(C, "CLUSTER_K", 16)
CLUSTER_MIN_OBS = getattr(C, "CLUSTER_MIN_OBS", 150)
MIN_GROUP_SIZE = getattr(C, "MIN_GROUP_SIZE", 4)

SECTOR_COLS = ["grp", "grp_size", "sec_cmf", "sec_acc", "sec_ret5", "sec_ret20", "lead_gap", "own_rel5", "sector_score"]


def _fit_clusters(resid_window: pd.DataFrame, k: int = CLUSTER_K) -> dict:
    """resid_window: (gün x hisse) piyasadan arındırılmış getiriler. {ticker: küme_no}"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    good = resid_window.columns[resid_window.notna().sum() >= min(CLUSTER_MIN_OBS, int(len(resid_window) * 0.6))]
    if len(good) < max(2 * k, 20):
        return {}
    cc = resid_window[good].corr(min_periods=max(60, int(len(resid_window) * 0.4))).fillna(0.0).values
    d = np.clip(1.0 - cc, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(z, k, criterion="maxclust")
    return dict(zip(good, lab.astype(int)))


def refit_dates(dates: pd.DatetimeIndex) -> list:
    """Her ayın ilk işlem günü (takvime sabit)."""
    s = pd.Series(dates, index=dates)
    return list(s.groupby([dates.year, dates.month]).min().values)


def point_in_time_clusters(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Uzun format (tarih, ticker, grp). Her ay yalnızca o aydan ÖNCEKİ verilerle kurulur."""
    if close_wide is None or close_wide.empty:
        return pd.DataFrame(columns=["tarih", "ticker", "grp"])
    c = close_wide.sort_index()
    r = c.pct_change()
    r = r.where(r.abs() <= C.MAX_VALID_DAILY_MOVE)
    resid = r.sub(r.median(axis=1), axis=0)
    dates = c.index
    pos = {d: i for i, d in enumerate(dates)}
    rows = []
    refits = refit_dates(dates)
    for j, rd in enumerate(refits):
        i = pos[pd.Timestamp(rd)]
        if i < CLUSTER_LOOKBACK * 0.8:
            continue
        mapping = _fit_clusters(resid.iloc[max(0, i - CLUSTER_LOOKBACK):i])
        if not mapping:
            continue
        end = pos[pd.Timestamp(refits[j + 1])] if j + 1 < len(refits) else len(dates)
        span = dates[i:end]
        tick = np.array(list(mapping.keys()))
        grp = np.array(list(mapping.values()))
        rows.append(pd.DataFrame({"tarih": np.repeat(span.values, len(tick)),
                                  "ticker": np.tile(tick, len(span)), "grp": np.tile(grp, len(span))}))
    if not rows:
        return pd.DataFrame(columns=["tarih", "ticker", "grp"])
    return pd.concat(rows, ignore_index=True)


def add_sector_features(df: pd.DataFrame, close_wide: pd.DataFrame) -> pd.DataFrame:
    """features.build_features çıktısına grup/sektör akış özelliklerini ekler."""
    if df is None or df.empty:
        return df
    out = df.drop(columns=[c for c in SECTOR_COLS if c in df.columns])
    cl = point_in_time_clusters(close_wide)
    ret5 = (close_wide / close_wide.shift(5) - 1.0) * 100.0
    ret20 = (close_wide / close_wide.shift(20) - 1.0) * 100.0
    extra = pd.DataFrame({"ret5": ret5.stack(), "ret20": ret20.stack()})
    extra.index.names = ["tarih", "ticker"]
    out = out.merge(extra.reset_index(), on=["tarih", "ticker"], how="left")
    out = out.merge(cl, on=["tarih", "ticker"], how="left")

    has = out["grp"].notna()
    key = [out["tarih"], out["grp"]]
    out["grp_size"] = out.groupby(key)["ticker"].transform("count").where(has)
    ok = has & (out["grp_size"] >= MIN_GROUP_SIZE)

    mkt5 = out.groupby("tarih")["ret5"].transform("median")
    mkt20 = out.groupby("tarih")["ret20"].transform("median")
    acc = ((out["cmf20"] > 0) & (out["z_vol"] > 0.5)).astype(float)
    out["sec_cmf"] = out.groupby(key)["cmf20"].transform("mean")
    out["sec_acc"] = acc.groupby(key).transform("mean")
    sec5 = out.groupby(key)["ret5"].transform("median")
    out["sec_ret5"] = sec5 - mkt5
    out["sec_ret20"] = out.groupby(key)["ret20"].transform("median") - mkt20
    out["own_rel5"] = out["ret5"] - sec5
    liq_rank = out.groupby(key)["liq20"].rank(ascending=False)
    lead = out["ret5"].where(liq_rank <= 3).groupby(key).transform("median")
    out["lead_gap"] = lead - out["ret5"]
    for col in ["sec_cmf", "sec_acc", "sec_ret5", "sec_ret20", "own_rel5", "lead_gap"]:
        out[col] = out[col].where(ok)

    def rk(s):
        return s.groupby(out["tarih"]).rank(pct=True) * 100.0

    score = (0.45 * rk(out["sec_cmf"]) + 0.20 * rk(out["sec_ret20"]) + 0.15 * rk(out["sec_acc"])
             + 0.10 * rk(out["lead_gap"]) + 0.10 * rk(-out["own_rel5"]))
    out["sector_score"] = score.where(ok).fillna(50.0)
    return out.drop(columns=["ret5", "ret20"])


def group_names(day_df: pd.DataFrame, official: dict | None = None, n_leaders: int = 3) -> dict:
    """Küme -> okunabilir isim: en likit üyeler (+ SP'de baskın resmi sektör)."""
    names = {}
    if day_df is None or day_df.empty or "grp" not in day_df.columns:
        return names
    for grp, part in day_df[day_df["grp"].notna()].groupby("grp"):
        leaders = part.sort_values("liq20", ascending=False)["ticker"].head(n_leaders).tolist()
        label = ", ".join(leaders)
        if official:
            secs = part["ticker"].map(official).dropna()
            if len(secs):
                top, share = secs.value_counts().index[0], secs.value_counts().iloc[0] / len(secs)
                if share >= 0.4:
                    label = f"{top} · {label}"
        names[int(grp)] = label
    return names


def sector_board(day_df: pd.DataFrame, official: dict | None = None, top: int = 5, bottom: int = 3) -> pd.DataFrame:
    """Günlük grup akış panosu (en güçlü birikim ve en belirgin dağıtım grupları)."""
    if day_df is None or day_df.empty or "grp" not in day_df.columns:
        return pd.DataFrame()
    d = day_df[day_df["grp"].notna() & (day_df["grp_size"] >= MIN_GROUP_SIZE)]
    if d.empty:
        return pd.DataFrame()
    b = d.groupby("grp").agg(n=("ticker", "count"), sec_cmf=("sec_cmf", "first"), sec_acc=("sec_acc", "first"),
                             sec_ret5=("sec_ret5", "first"), sec_ret20=("sec_ret20", "first"),
                             score=("sector_score", "median")).reset_index()
    names = group_names(d, official)
    b["name"] = b["grp"].astype(int).map(names)
    # "sessiz birikim": birikim yüksek ama 5g göreli getiri henüz düşük -> hareket öncesi toplama
    b["stealth"] = (b["sec_cmf"].rank(pct=True) - b["sec_ret5"].rank(pct=True)).round(2)
    b = b.sort_values("score", ascending=False)
    # Mutlak birikim (CMF>0) yoksa "görece güçlü" denir; zayıf piyasada yanıltıcı etiket verilmez
    b["side"] = np.where(b["sec_cmf"] > 0, "BİRİKİM", "GÖRECE GÜÇLÜ")
    tail = b.tail(bottom).copy()
    tail["side"] = "DAĞITIM"
    return pd.concat([b.head(top), tail], ignore_index=True)
