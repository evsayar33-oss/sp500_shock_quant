"""S&P 500 Adaptive Meta-Engine v2 — NEW YORK KAPANIŞI SONRASI tarama (günlük).

Akış
 1) Evren: S&P 500 bileşenleri (Wikipedia) + TradingView kesiti (bugünün barı yedeği, sektör) [ücretsiz]
 2) Panel güncelle (yfinance, bölünme kontrollü) + makro seriler (VIX, kredi, faiz, dolar)       [ücretsiz]
 3) Özellikler + bilanço karartması -> rejim (kesitsel + ABD makro)
 4) Skor (öğrenilmiş rejim profilleri) -> otonomi koruması -> bilanço kontrolü
 5) Portföy: vol-hedefli boyut + korelasyon filtresi + brüt limit
 6) Defter: bekleyen giriş (T+1 açılış), açık pozisyon takibi, kesin tarihli etiketleme
 7) Telegram raporu

Sinyal T günü kapanışta üretilir; giriş ertesi gün AÇILIŞTA yapılır. Backtest etiketi ile birebir.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config as C
import exits
import health
import live_book as LB
import report
import price_history as HIST
from autonomy_guard import evaluate_autonomy_guard
from price_history import (append_live_bar, load_earnings, load_members, load_sectors, merge_universe,
                           refresh_constituents, update_earnings, update_incremental, update_macro)
from features import attach_regime, build_features, correlation_matrix
from meta_label import predict as meta_predict, select_with_meta
from portfolio import build_portfolio
from sector_flow import group_names, sector_board
from regime import compute_macro_frame
from sp_engine import entry_status, gecmis_veriyi_yukle, score_frame, stars_for
from sp_fetcher import get_sp500_raw_data
from sp_learner import (_label_rows, active_profiles, load_ai_state, load_signal_history, log_shock_signals,
                           save_ai_state, update_realized_shock_returns)

LIVE_LOOKBACK_DAYS = 300   # kümeler için >= CLUSTER_LOOKBACK + 1 ay


# ------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------
def send_telegram_message(message):
    report.send(message)


# ------------------------------------------------------------------
# Veri
# ------------------------------------------------------------------
def fetch_live_snapshot():
    try:
        df = get_sp500_raw_data()
        if df is not None and len(df) > 10:
            return df
    except Exception as exc:
        print(f"TradingView hatası: {exc}")
    return pd.DataFrame()


def live_bar_is_fresh(panel, live):
    """Tatil/hafta sonu elle çalıştırmada TV önceki seansı döndürür; panelin son günüyle aynıysa ekleme."""
    if live.empty or panel.empty:
        return not live.empty
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    if now.weekday() >= 5 or (now.hour, now.minute) < (16, 15):
        return False
    last = panel[panel["tarih"] == panel["tarih"].max()][["ticker", "close"]]
    m = live.merge(last, on="ticker", suffixes=("", "_p"))
    if len(m) < 20:
        return True
    same = (np.abs(m["close"] / m["close_p"] - 1.0) < 1e-6).mean()
    return same < 0.8


def data_quality(panel, live_ok):
    if panel.empty:
        return 0.0, "PANEL_YOK"
    last = panel["tarih"].max()
    lag = int(np.busday_count(last.date(), pd.Timestamp.now(tz="America/New_York").date()))
    n_last = int((panel["tarih"] == last).sum())
    n_prev = panel[panel["tarih"] < last].groupby("tarih").size().tail(20).median() if panel["tarih"].nunique() > 1 else n_last
    coverage = n_last / max(n_prev, 1)
    score = 100.0
    if lag > 1:
        score -= min(20.0 * (lag - 1), 60.0)
    if coverage < 0.8:
        score -= (0.8 - coverage) * 100.0
    if not live_ok:
        score -= 5.0
    return float(np.clip(score, 0, 100)), f"son_gün={last.date()} gecikme={lag}g kapsam=%{coverage * 100:.0f}"


# ------------------------------------------------------------------
# Bilanço
# ------------------------------------------------------------------
def check_earnings_risk(ticker):
    """Canlı ikinci kontrol: yfinance takvimi (bilanço dosyasında eksik tarih varsa yakalar)."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        ed = cal.get("Earnings Date") if hasattr(cal, "get") else None
        if ed is None:
            return False, ""
        ed = ed if isinstance(ed, (list, tuple, np.ndarray)) else [ed]
        for d in ed:
            if pd.notna(d):
                dd = pd.Timestamp(d).date()
                diff = (dd - datetime.now().date()).days
                if 0 <= diff <= C.HORIZON + 3:
                    return True, f"{dd.strftime('%d.%m')} ({diff} gün)"
    except Exception:
        pass
    return False, ""


# ------------------------------------------------------------------
# Defter (ledger) v2
# ------------------------------------------------------------------
def load_ledger():
    """v3 defter (live_book). Denetçi ve geriye uyum için."""
    return LB.load_ledger()


def retry_missing_last_bar(panel, wait_sec=90):
    """Teşhis bulgusu: kapanış sonrası ilk dakikalarda Yahoo bazı hisselerin son barını henüz yayınlamıyor.
    Son gün kapsamı düşükse bekleyip yalnızca eksik hisseleri yeniden indirir (tek deneme)."""
    import time
    if panel is None or panel.empty:
        return panel, 0
    last = panel["tarih"].max()
    per_day = panel.groupby("tarih").size()
    med = per_day.tail(21).iloc[:-1].median() if len(per_day) > 2 else per_day.iloc[-1]
    if per_day.iloc[-1] >= 0.95 * med:
        return panel, 0
    recent = set(panel.loc[panel["tarih"] >= last - pd.Timedelta(days=10), "ticker"])
    missing = sorted(recent - set(panel.loc[panel["tarih"] == last, "ticker"]))
    if not missing:
        return panel, 0
    print(f"⏳ Son gün kapsamı düşük ({per_day.iloc[-1]}/{med:.0f}); {len(missing)} hisse {wait_sec} sn sonra yeniden denenecek")
    time.sleep(wait_sec)
    fresh = HIST.download_ohlcv(missing, period="5d")
    fresh = fresh[fresh["tarih"] == last] if not fresh.empty else fresh
    if fresh.empty:
        return panel, 0
    out = pd.concat([panel, fresh], ignore_index=True).drop_duplicates(["tarih", "ticker"], keep="last")
    out = out.sort_values(["tarih", "ticker"]).reset_index(drop=True)
    HIST.save_panel(out, months={pd.Timestamp(last).strftime("%Y_%m")})
    print(f"✅ {len(fresh)} eksik bar tamamlandı")
    return out, int(len(fresh))


def main():
    print(f"[{datetime.now():%H:%M:%S}] === S&P 500 Meta-Engine v3 ===")
    live = fetch_live_snapshot()
    import os as _os
    import json as _json
    uni = {}
    try:
        uni = _json.load(open(C.UNIVERSE_FILE, encoding="utf-8"))
    except Exception:
        pass
    if not uni.get("members") or str(uni.get("updated", ""))[:7] != pd.Timestamp.now().strftime("%Y-%m"):
        refresh_constituents()  # ayda en az bir kez bileşen listesi yenilenir
    if not live.empty and "sector" in live.columns:
        members = load_members()
        known = load_sectors()  # Wikipedia GICS adları korunur; TradingView yalnızca eksikleri doldurur
        merge_universe([], sectors={t: s for t, s in zip(live["ticker"], live["sector"])
                                    if t in members and s and not known.get(t)})

    panel = update_incremental()
    macro_raw = update_macro()
    macro_frame = compute_macro_frame(macro_raw)
    panel, n_retry = retry_missing_last_bar(panel)
    yahoo_last = str(pd.Timestamp(panel["tarih"].max()).date()) if not panel.empty else None
    n_before = len(panel)
    if panel.empty:
        send_telegram_message("⚠️ S&P 500 v2: geçmiş panel oluşturulamadı (yfinance erişimi). Tarama atlandı.")
        return
    live_ok = live_bar_is_fresh(panel, live)
    if live_ok:
        members = load_members()
        panel = append_live_bar(panel, live[live["ticker"].isin(members)],
                                pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize())
    earnings = load_earnings()
    if earnings.empty:
        earnings = update_earnings()
    tv_appended = len(panel) - n_before
    dq, dq_text = data_quality(panel, not live.empty)
    print(f"Veri kalitesi {dq:.0f} | {dq_text}")

    recent_days = sorted(panel["tarih"].unique())[-LIVE_LOOKBACK_DAYS:]
    feat = build_features(panel[panel["tarih"].isin(recent_days)], with_labels=False, earnings=earnings)
    day = feat["tarih"].max()
    today = attach_regime(feat[feat["tarih"] == day], macro_frame)
    # Yalnızca güncel endeks üyeleri işlem adayıdır (eski üyeler geçmişte kalır, kesitsel sıralamaya katılır)
    members = load_members()
    today.loc[~today["ticker"].isin(members), "eligible"] = False
    sectors = load_sectors()
    today["sector"] = today["ticker"].map(sectors)
    if today.empty:
        print("Bugün için özellik yok.")
        return

    from regime import classify_market_regime, combine_regime, macro_snapshot
    cross = classify_market_regime(today[~today["is_illiquid"]])
    regime = combine_regime(cross, macro_snapshot(macro_frame, day))
    regime["day"] = day
    print(f"Rejim: {regime['label']} | makro {regime['macro_label']} stres {regime['macro_stress']}")

    state = load_ai_state()
    profiles = active_profiles(state)  # v1 durum dosyası burada otomatik arşivlenir
    offset = float(state.get("win_rate_optimizer", {}).get("active_threshold", 75.0)) - 75.0
    offset = float(np.clip(offset, -6.0, 10.0))

    # Etiketleri güncelle (performans kayması için)
    sig_hist = update_realized_shock_returns(panel)
    # Performans kayması: v3 defterde KAPANMIŞ canlı işlemlerin net getirisi (çıkış kuralı dahil)
    _l3 = LB.v3_rows(LB.load_ledger())
    perf = pd.to_numeric(_l3.loc[_l3["status"] == "CLOSED", "net_ret"], errors="coerce").dropna() if not _l3.empty else None

    scored = score_frame(today, profiles, threshold_offset=offset)
    guard = evaluate_autonomy_guard(state, features=scored, regime={"label": regime["label"]},
                                    regime_confidence=float(regime["confidence"]), performance_returns=perf,
                                    data_quality_score=dq, row_count=len(scored), min_rows=100, project="sp500_shock")
    scored = score_frame(today, profiles, threshold_offset=offset,
                         extra_threshold_add=float(guard.get("signal_threshold_add", 0.0)))
    if guard.get("block_new_entries"):
        scored["effective_min_score"] = 101.0
    scored["entry_status"] = [entry_status(r) for _, r in scored.iterrows()]
    official = load_sectors()
    gnames = group_names(today, official)
    board = sector_board(today, official)
    try:
        os.makedirs(C.DATA_DIR, exist_ok=True)
        board.to_json(os.path.join(C.DATA_DIR, "sector_board.json"), orient="records", force_ascii=False, indent=1)
    except Exception as exc:
        print(f"Pano kaydedilemedi: {exc}")
    scored["stars"] = [stars_for(r) for _, r in scored.iterrows()]
    scored = scored.sort_values(["shock_score", "risk_adjusted_score"], ascending=False).reset_index(drop=True)

    # Kazanma olasılığı filtresi (meta-etiket): yalnızca denetim OOS kanıtla açtıysa kullanılır
    ml = state.get("meta_label") or {}
    model = ml.get("model") if ml.get("enabled") else None
    if model:
        scored["p_win"] = np.nan
        elig_idx = scored.index[scored["eligible"]]
        if len(elig_idx):
            scored.loc[elig_idx, "p_win"] = meta_predict(model, scored.loc[elig_idx]).values
        cands = select_with_meta(scored, model).copy()
        if guard.get("block_new_entries"):
            cands = cands.iloc[0:0]
    else:
        cands = scored[scored["shock_score"] >= scored["effective_min_score"]].head(C.TOP_K_PER_DAY).copy()
    for idx, r in cands.iterrows():
        has, when = check_earnings_risk(r["ticker"])
        if has:
            cands.at[idx, "shock_score"] = 0.0
            cands.at[idx, "entry_status"] = f"🚨 BİLANÇO RİSKİ ({when})"
    cands = cands[cands["shock_score"] > 0]
    exposure = float(guard.get("exposure_multiplier", 1.0)) * float(regime["macro_exposure_mult"])
    regime["exposure_mult"] = exposure
    corr = correlation_matrix(panel, cands["ticker"].tolist(), end_date=day)
    port = build_portfolio(cands, corr, exposure_mult=exposure)

    # Varsayılan tahsis metni (portföy dışı satırlar için)
    scored["allocation"] = "İşlem Açma"
    scored["weight_pct"] = 0.0
    if not port.empty:
        amap = dict(zip(port["ticker"], port["allocation"]))
        wmap = dict(zip(port["ticker"], port["weight_pct"]))
        scored["allocation"] = scored["ticker"].map(amap).fillna("İşlem Açma")
        scored["weight_pct"] = scored["ticker"].map(wmap).fillna(0.0)

    # Shadow (aday) profil ile gölge skor
    shadow_df = None
    sh = state.get("meta_engine", {}).get("shadow", {}).get("profiles")
    if isinstance(sh, dict) and sh:
        shadow_df = score_frame(today, sh, threshold_offset=offset).sort_values("watch_score", ascending=False).head(8)

    log_shock_signals(scored.head(10), regime_snapshot=regime, shadow_df=shadow_df)

    # v3 defter: açık pozisyonları panelden yeniden hesapla (TP1/başabaş/TP2/stop/süre), yeni sinyalleri ekle
    ledger = LB.load_ledger()
    ledger, events = LB.update_book(ledger, panel)
    port_rec = port.copy() if not port.empty else port
    if not port_rec.empty:
        port_rec["grp_name"] = port_rec["grp"].map(lambda g: gnames.get(int(g)) if g == g and g is not None else None)
    ledger = LB.record_signals(ledger, port_rec, day)
    ledger.to_csv(C.LEDGER_FILE, index=False)
    positions = LB.positions_view(ledger, panel)
    live_summary = LB.build_tables(ledger, state.get("backtest_summary"))

    # Otomatik yeniden eğitim tetikleyicileri + sistem sağlığı
    health.retrain_triggers(state, regime["label"], guard, live_summary)
    health.heartbeat(state, "scan")
    hl = health.evaluate(state, panel, macro_raw, {
        "live_rows": int(len(live)), "tv_appended": int(max(tv_appended, 0)), "yahoo_last": yahoo_last,
        "yahoo_rows": int((panel["tarih"] == panel["tarih"].max()).sum()) - int(max(tv_appended, 0)),
        "dq": dq, "guard_mode": guard.get("mode"), "live_summary": live_summary,
        "calibration": health.calibration(LB.v3_rows(ledger)) if model else None})

    # app.py için günlük kesit geçmişi (120 gün)
    scored["grp_name"] = scored["grp"].map(lambda g: gnames.get(int(g)) if g == g and g is not None else None)
    hist = gecmis_veriyi_yukle()
    snap = scored.copy()
    snap["tarih"] = pd.Timestamp(day)
    if not hist.empty and "tarih" in hist.columns:
        hist = hist[pd.to_datetime(hist["tarih"], errors="coerce").dt.normalize() != pd.Timestamp(day)]
        snap = pd.concat([hist, snap], ignore_index=True, sort=False)
    snap["tarih"] = pd.to_datetime(snap["tarih"], errors="coerce")
    snap = snap[snap["tarih"] >= pd.Timestamp(day) - pd.Timedelta(days=120)]
    snap.to_csv(C.GECMIS_DOSYA, index=False)

    state["last_scan"] = {"day": str(pd.Timestamp(day).date()), "regime": {k: v for k, v in regime.items() if k != "day"},
                          "data_quality": dq, "threshold_offset": offset, "n_candidates": int(len(cands)),
                          "n_positions": int((port["weight_pct"] > 0).sum()) if not port.empty else 0}
    state["status"] = f"🧠 META v2 | {regime['label']} | makro {regime['macro_label']} | guard {guard.get('mode')}"
    save_ai_state(state)

    picks = []
    if not port.empty:
        for _, r in port[port["weight_pct"] > 0].iterrows():
            g = r.get("grp")
            picks.append({"ticker": r["ticker"], "close": r["close"], "change": r.get("change_%"),
                          "score": r["shock_score"], "thr": r["effective_min_score"], "p_win": r.get("p_win"),
                          "weight": r["weight_pct"], "sec_cmf": r.get("sec_cmf"),
                          **exits.levels(float(r["close"]), float(r["atr"]) / float(r["close"])),
                          "group": gnames.get(int(g)) if g == g and g is not None else None})
    watch = []
    if not picks:
        w = scored[scored["eligible"]].sort_values("watch_score", ascending=False).head(3)
        watch = [{"ticker": r.ticker, "score": r.watch_score, "thr": r.effective_min_score,
                  "p_win": getattr(r, "p_win", None) if model else None, "q": (model or {}).get("q")}
                 for r in w.itertuples() if r.watch_score >= r.effective_min_score - 8]
    report.send(report.scan_message({
        "day": day, "regime": regime, "guard": guard, "exposure": exposure, "picks": picks, "watch": watch,
        "positions": positions, "board": board, "scorecard": state.get("backtest_summary"),
        "meta_on": bool(model), "dq": dq, "events": events, "live": live_summary, "health": hl,
        "week_end": pd.Timestamp(day).weekday() == 4}))
    print("Tarama tamamlandı.")


if __name__ == "__main__":
    main()
