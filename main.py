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
from autonomy_guard import evaluate_autonomy_guard
from price_history import (append_live_bar, load_earnings, load_members, load_sectors, merge_universe,
                           refresh_constituents, update_earnings, update_incremental, update_macro)
from features import attach_regime, build_features, correlation_matrix
from portfolio import build_portfolio
from regime import compute_macro_frame
from sp_engine import entry_status, gecmis_veriyi_yukle, score_frame, stars_for
from sp_fetcher import get_sp500_raw_data
from sp_learner import (_label_rows, active_profiles, load_ai_state, load_signal_history, log_shock_signals,
                           save_ai_state, update_realized_shock_returns)

LIVE_LOOKBACK_DAYS = 220


# ------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------
def send_telegram_message(message):
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("CHAT_ID")
    if not token or not chat_id:
        print(message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    parts, cur = [], ""
    for block in message.split("\n\n"):
        if len(cur) + len(block) + 2 > 3800 and cur.strip():
            parts.append(cur.strip())
            cur = ""
        cur += block + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    for msg in parts:
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            res = requests.post(url, json=payload, timeout=15)
            if not res.json().get("ok"):
                payload.pop("parse_mode", None)
                requests.post(url, json=payload, timeout=15)
        except Exception as exc:
            print(f"Telegram hatası: {exc}")


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
    if not os.path.exists(C.LEDGER_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(C.LEDGER_FILE)
    except Exception:
        return pd.DataFrame()


def update_ledger_from_panel(ledger, panel):
    if ledger.empty:
        return ledger
    for c in ("label_version", "entry_date", "entry_price", "cost_rt", "net_ret_5d", "exit_date",
              "price_d1", "price_d3", "price_d5", "return_d1", "return_d3", "return_d5", "is_completed"):
        if c not in ledger.columns:
            ledger[c] = np.nan
    ledger["is_completed"] = pd.to_numeric(ledger["is_completed"], errors="coerce").fillna(0).astype(int)
    for c in ("entry_date", "exit_date", "date"):
        ledger[c] = ledger[c].astype(object)
    todo = ledger[(pd.to_numeric(ledger["label_version"], errors="coerce") == 2) & (ledger["is_completed"] == 0)]
    labels = _label_rows(todo, "date", "ticker", panel, liq_col="liq20")
    for idx, lab in labels.items():
        if "entry_price" not in lab:
            continue
        ledger.at[idx, "entry_date"] = lab["entry_date"]
        ledger.at[idx, "entry_price"] = lab["entry_price"]
        ledger.at[idx, "cost_rt"] = lab["cost_rt"]
        for k in (1, 3, C.HORIZON):
            if f"close_d{k}" in lab:
                ledger.at[idx, f"price_d{k}"] = lab[f"close_d{k}"]
                ledger.at[idx, f"return_d{k}"] = lab[f"gross_d{k}"]
        if f"gross_d{C.HORIZON}" in lab:
            ledger.at[idx, "net_ret_5d"] = round(lab[f"gross_d{C.HORIZON}"] - lab["cost_rt"], 3)
            ledger.at[idx, "exit_date"] = lab.get("exit_date")
            ledger.at[idx, "is_completed"] = 1
    return ledger


def record_ledger_entries(ledger, portfolio_df, day):
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    if not ledger.empty and "date" in ledger.columns:
        lv = pd.to_numeric(ledger.get("label_version"), errors="coerce")
        ledger = ledger[~((ledger["date"].astype(str) == day_str) & (lv == 2))]  # aynı gün yeniden çalıştırma
    if portfolio_df is None or portfolio_df.empty:
        return ledger
    picks = portfolio_df[portfolio_df["weight_pct"] > 0]
    if picks.empty:
        return ledger
    rows = []
    for _, r in picks.iterrows():
        rows.append({
            "date": day_str, "ticker": r["ticker"], "label_version": 2, "signal_close": r["close"],
            "entry_date": np.nan, "entry_price": np.nan, "weight_pct": r["weight_pct"], "atr": r.get("atr"),
            "stop_atr_mult": C.STOP_ATR, "initial_score": r["shock_score"],
            "effective_min_score": r["effective_min_score"], "regime": r.get("meta_regime"),
            "regime_confidence": r.get("meta_regime_confidence"), "macro_label": r.get("macro_label"),
            "macro_stress": r.get("macro_stress"), "liq20": r.get("liq20"), "volatility": r.get("volatility"),
            "z_vol": r.get("z_vol"), "z_range": r.get("z_range"), "z_flow": r.get("z_flow"),
            "z_lambda": r.get("z_lambda"), "cmf20": r.get("cmf20"), "resilience_score": r.get("resilience_score"),
            "excess_return": r.get("excess_return"), "flow_score": r.get("flow_score"),
            "event_score": r.get("event_score"), "activity_score": r.get("activity_score"),
            "liquidity_score": r.get("liquidity_score"), "overnight_risk": r.get("overnight_risk"),
            "entry_status": r.get("entry_status"), "sector": r.get("sector"), "is_completed": 0,
        })
    new = pd.DataFrame(rows)
    return pd.concat([ledger, new], ignore_index=True, sort=False)


def exit_engine_text(ledger, panel):
    if ledger.empty or panel.empty:
        return ""
    lv = pd.to_numeric(ledger.get("label_version"), errors="coerce")
    open_pos = ledger[(lv == 2) & (ledger["is_completed"] == 0)]
    if open_pos.empty:
        return ""
    last_day = panel["tarih"].max()
    today = panel[panel["tarih"] == last_day].set_index("ticker")
    cal = pd.DatetimeIndex(sorted(panel["tarih"].unique()))
    lines = []
    for _, r in open_pos.iterrows():
        t = r["ticker"]
        if pd.isna(r.get("entry_price")):
            lines.append(f"⏳ <b>#{t}</b> — giriş bekleniyor (sinyal {r['date']}, sonraki seans AÇILIŞINDA al, ağırlık %{float(r.get('weight_pct', 0)):.1f})")
            continue
        if t not in today.index:
            continue
        entry = float(r["entry_price"])
        cur = float(today.at[t, "close"])
        low = float(today.at[t, "low"])
        atr = float(r.get("atr") or 0.0)
        stop = entry - C.STOP_ATR * atr if atr > 0 else entry * 0.93
        held = int(((cal >= pd.Timestamp(r["entry_date"])) & (cal <= last_day)).sum())
        pnl = (cur / entry - 1.0) * 100.0
        if low <= stop:
            lines.append(f"🚨 <b>#{t} FELAKET STOPU</b> (${stop:.2f}) kırıldı | K/Z %{pnl:+.2f} → sonraki açılışta çık")
        elif held >= C.HORIZON:
            lines.append(f"⏰ <b>#{t} VADE DOLDU</b> ({held}/{C.HORIZON}) | K/Z %{pnl:+.2f} → kapanışta çıkılmış sayılır")
        else:
            lines.append(f"🟢 <b>#{t}</b> {held}/{C.HORIZON}. gün | K/Z %{pnl:+.2f} | stop ${stop:.2f}")
    if not lines:
        return ""
    return "🛡️ <b>AÇIK / BEKLEYEN POZİSYONLAR</b>\n" + "\n".join(lines) + "\n━━━━━━━━━━━━━━━━━━━━\n\n"


# ------------------------------------------------------------------
# Rapor
# ------------------------------------------------------------------
def format_report(port, scored_today, regime, guard, state, dq_text, exit_text):
    wf = state.get("backtest_summary", {})
    msg = exit_text
    msg += "🗽 <b>S&P 500 ADAPTIVE META-ENGINE v2</b>\n"
    msg += f"🗓 <i>{pd.Timestamp(regime['day']).strftime('%Y-%m-%d')} NY kapanış | giriş: sonraki seans açılışı | çıkış: T+{C.HORIZON} kapanış</i>\n"
    msg += (f"🌐 <b>Rejim:</b> {regime['label']} (kesitsel {regime.get('cross_label')}, güven %{regime['confidence'] * 100:.0f})\n"
            f"🌍 <b>Makro:</b> {regime.get('macro_label')} | stres {regime.get('macro_stress', 0):.2f} | "
            f"VIX {(regime.get('vix') or 0):.1f} (vade {(regime.get('vix_term') or 0):.2f}) | "
            f"HYG-LQD 20g %{(regime.get('credit_rel20') or 0):+.1f} | SPY/200g %{(regime.get('spy_trend200') or 0):+.1f}\n")
    msg += (f"🛡️ <b>Otonomi:</b> {guard.get('mode')} ({guard.get('reason')}) | maruziyet x{regime['exposure_mult']:.2f}\n")
    if wf:
        msg += (f"📊 <b>OOS doğrulama:</b> N={wf.get('n', 0)} | WR %{wf.get('win_rate', 0):.1f} (LCB %{wf.get('wilson_lcb', 0):.1f}) | "
                f"net ort %{wf.get('avg_return', 0):+.2f} | PF {wf.get('profit_factor', 0):.2f} | t={wf.get('cohort_t', 0):.1f}\n")
    msg += f"🔧 <i>Veri: {dq_text}</i>\n━━━━━━━━━━━━━━━━━━━━\n\n"

    picks = port[port["weight_pct"] > 0] if port is not None and not port.empty else pd.DataFrame()
    if picks.empty:
        eff = float(scored_today["effective_min_score"].median()) if not scored_today.empty else 0
        watch = scored_today[scored_today["watch_score"] >= eff - 5].sort_values("watch_score", ascending=False).head(5)
        if not watch.empty:
            msg += "🔎 <b>İZLEME (eşik altı):</b>\n" + "".join(
                f"• #{r.ticker} | skor {r.watch_score:.1f} / eşik {r.effective_min_score:.1f}\n" for r in watch.itertuples())
        return msg + "\n🛡️ <i>Bugün giriş kriterlerini geçen aday yok.</i>"

    for _, r in picks.iterrows():
        msg += f"🚀 <b>#{r['ticker']}</b> ── <b>{r['shock_score']:.1f}</b> / eşik {r['effective_min_score']:.1f} ({r['stars']})\n"
        msg += (f"• Günlük %{r['change_%']:+.2f} | piyasa üstü %{r.get('excess_return', 0):+.2f} | "
                f"RVOL {r.get('rvol', 1):.2f}x | zVol {r.get('z_vol', 0):+.1f}σ\n")
        msg += (f"• Olay {r['event_score']:.0f} | Birikim {r['flow_score']:.0f} (CMF {r.get('cmf20', 0):+.2f}) | "
                f"Dayanıklılık {r['resilience_score']:.0f} | Gap-risk {r['overnight_risk']:.0f}\n")
        msg += f"• ${r['close']:.2f} | {r.get('sector') or '-'} | Giriş: <i>{r['entry_status']}</i> | Felaket stop ≈ ${r['stop_price']:.2f}\n"
        msg += f"💰 <b>Ağırlık: {r['allocation']}</b>\n\n"
    skipped = port[(port["weight_pct"] <= 0) & (port["skip_reason"] != "")]
    if not skipped.empty:
        msg += "↪️ <i>Elenen: " + ", ".join(f"#{a} ({b})" for a, b in zip(skipped["ticker"], skipped["skip_reason"])) + "</i>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n🎯 <i>{len(picks)} pozisyon | brüt %{picks['weight_pct'].sum():.1f}</i>"
    return msg


# ------------------------------------------------------------------
# Ana akış
# ------------------------------------------------------------------
def main():
    print(f"[{datetime.now():%H:%M:%S}] === S&P 500 Adaptive Meta-Engine v2 ===")
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
        merge_universe([], sectors={t: s for t, s in zip(live["ticker"], live["sector"]) if t in members and s})

    panel = update_incremental()
    macro_frame = compute_macro_frame(update_macro())
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
    perf = None
    if not sig_hist.empty and "realized_5d" in sig_hist.columns:
        v2 = sig_hist[(pd.to_numeric(sig_hist.get("label_version"), errors="coerce") == 2)
                      & (sig_hist.get("model_variant") == "active")]
        perf = pd.to_numeric(v2["realized_5d"], errors="coerce").dropna()

    scored = score_frame(today, profiles, threshold_offset=offset)
    guard = evaluate_autonomy_guard(state, features=scored, regime={"label": regime["label"]},
                                    regime_confidence=float(regime["confidence"]), performance_returns=perf,
                                    data_quality_score=dq, row_count=len(scored), min_rows=100, project="sp500_shock")
    scored = score_frame(today, profiles, threshold_offset=offset,
                         extra_threshold_add=float(guard.get("signal_threshold_add", 0.0)))
    if guard.get("block_new_entries"):
        scored["effective_min_score"] = 101.0
    scored["entry_status"] = [entry_status(r) for _, r in scored.iterrows()]
    scored["stars"] = [stars_for(r) for _, r in scored.iterrows()]
    scored = scored.sort_values(["shock_score", "risk_adjusted_score"], ascending=False).reset_index(drop=True)

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

    ledger = load_ledger()
    ledger = update_ledger_from_panel(ledger, panel)
    ledger = record_ledger_entries(ledger, port, day)
    ledger.to_csv(C.LEDGER_FILE, index=False)
    exit_text = exit_engine_text(ledger, panel)

    # app.py için günlük kesit geçmişi (120 gün)
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

    send_telegram_message(format_report(port, scored, regime, guard, state, dq_text, exit_text))
    print("Tarama tamamlandı.")


if __name__ == "__main__":
    main()
