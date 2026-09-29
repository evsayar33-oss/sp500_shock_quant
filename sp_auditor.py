"""S&P 500 Adaptive Meta-Engine v2 — öğrenme ve denetim (kendini iyileştirme döngüsü).

Her çalıştırmada:
 1) 3 yıllık panelden özellik + NET etiket + bilanço karartması + rejim (kesitsel+makro) üretilir.
 2) Purged/embargo'lu walk-forward: aday ÖĞRENME PROSEDÜRÜ vs aktif profil, aynı OOS dilimlerinde.
 3) Terfi yalnızca OOS kanıtla (Wilson LCB / net ortalama / PF korumaları); aksi halde shadow.
 4) Win-rate optimizer global eşik ofsetini OOS havuzda (embargo'lu) ayarlar.
 5) Canlı defter kötüleşirse stable profile geri dönüş (rollback).
 6) Rapor: data/backtest_report.json + Telegram özeti.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
from autonomy_guard import evaluate_autonomy_guard
from price_history import load_earnings, load_macro, load_panel
from features import attach_regime, build_features
from regime import compute_macro_frame
from sp_learner import (active_profiles, ensure_meta_state, learn_profiles, live_rollback_needed, load_ai_state,
                           load_signal_history, promotion_decision, save_ai_state, walk_forward)
from win_rate_optimizer import optimize_win_rate, summary as winrate_summary


def send_telegram_audit(message):
    import requests
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("CHAT_ID")
    if not token or not chat_id:
        print(message)
        return
    payload = {"chat_id": chat_id, "text": message[:4000], "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        res = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=15)
        if not res.json().get("ok"):
            payload.pop("parse_mode", None)
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=15)
    except Exception as exc:
        print(f"Telegram audit hatası: {exc}")


def _slim(p):
    return {k: p.get(k) for k in ("weights", "signs", "sign_stats", "min_score", "regime", "version", "learned_edges",
                                  "regime_days", "threshold_source", "train_metrics") if k in p}


def build_research():
    panel = load_panel()
    if panel.empty:
        return pd.DataFrame(), panel
    feat = build_features(panel, with_labels=True, earnings=load_earnings())
    research = attach_regime(feat, compute_macro_frame(load_macro()))
    return research, panel


def run_audit():
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] S&P 500 v2 öğrenme denetimi başlıyor...")
    state = load_ai_state()
    meta = ensure_meta_state(state)
    research, panel = build_research()
    if research.empty:
        send_telegram_audit("⚠️ S&P 500 v2 denetim: panel yok (önce main.py / price_history.py çalışmalı).")
        save_ai_state(state)
        return

    active = active_profiles(state)
    offset = float(state.get("win_rate_optimizer", {}).get("active_threshold", 75.0)) - 75.0
    wf = walk_forward(research, active, offset=offset)
    decision, note = False, ""
    rollback = False

    if not wf.get("ok"):
        note = wf.get("reason", "WARMUP")
        meta["last_decision"] = {"promoted": False, "note": note, "timestamp": t0.isoformat(timespec="seconds")}
    else:
        decision, note = promotion_decision(wf["active"], wf["candidate"])
        final = learn_profiles(research)  # tüm etiketli veriyle nihai aday
        if decision:
            meta["stable_profiles"] = {r: _slim(p) for r, p in active.items()}
            meta["regime_profiles"] = {r: {**_slim(p), "promoted_at": t0.isoformat(timespec="seconds")} for r, p in final.items()}
            meta["promotion_count"] = int(meta.get("promotion_count", 0)) + 1
            meta["shadow"] = {}
        else:
            meta["shadow"] = {"profiles": {r: _slim(p) for r, p in final.items()},
                              "created_at": t0.isoformat(timespec="seconds"), "decision": note}

        # Win-rate optimizer: OOS aday havuzunda global eşik ofseti
        pool = wf["pool"].rename(columns={"net_ret": "realized_5d"})
        state = optimize_win_rate(state, pool, current_threshold=75.0 + offset)
        meta["winrate_optimizer_status"] = winrate_summary(state)

        state["backtest_summary"] = wf["candidate"] if decision else wf["active"]
        report = {"generated": t0.isoformat(timespec="seconds"), "days": wf["days"], "horizon": C.HORIZON,
                  "cost_model": {"commission_bps": C.COMMISSION_BPS, "slippage": "base+coef/sqrt(liq)"},
                  "candidate_oos": wf["candidate"], "active_oos": wf["active"], "template_oos": wf["template"],
                  "ic_oos": wf["ic"], "folds": wf["folds"], "decision": note, "promoted": decision}
        os.makedirs(C.DATA_DIR, exist_ok=True)
        with open(C.BACKTEST_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        keep = ["tarih", "ticker", "shock_score", "effective_min_score", "regime_label", "macro_label",
                "net_ret", "gross_ret", "cost_rt", "gap_next", "fold"]
        tr = wf["candidate_trades"]
        if not tr.empty:
            tr[[c for c in keep if c in tr.columns]].to_csv(C.OOS_TRADES_FILE, index=False, compression={"method": "gzip", "mtime": 0})

    # Canlı rollback
    from main import load_ledger
    rb, live_m = live_rollback_needed(load_ledger())
    if rb and meta.get("stable_profiles"):
        meta["regime_profiles"] = meta["stable_profiles"]
        meta["rollback_count"] = int(meta.get("rollback_count", 0)) + 1
        rollback = True
        note = "ROLLBACK | canlı defter güvenlik sınırını bozdu"
    state["live_metrics"] = live_m

    # Otonomi: performans kayması kapanışta da değerlendirilir
    sig = load_signal_history()
    perf = None
    if not sig.empty and "realized_5d" in sig.columns and "label_version" in sig.columns:
        perf = pd.to_numeric(sig.loc[pd.to_numeric(sig["label_version"], errors="coerce") == 2, "realized_5d"],
                             errors="coerce").dropna()
    last_reg = state.get("last_scan", {}).get("regime", {})
    evaluate_autonomy_guard(state, features=None, regime={"label": last_reg.get("label", "NORMAL")},
                            regime_confidence=float(last_reg.get("confidence", 0.5) or 0.5), performance_returns=perf,
                            data_quality_score=100.0, row_count=int(research["ticker"].nunique()), min_rows=100,
                            project="sp500_shock", stress_test_due=False)

    meta["last_decision"] = {"promoted": decision, "rollback": rollback, "note": note,
                             "timestamp": t0.isoformat(timespec="seconds")}
    save_ai_state(state)

    # Rapor
    rep = "🗽 <b>S&P 500 META-ENGINE v2 / ÖĞRENME DENETİMİ</b>\n"
    rep += f"🗓 <i>{t0:%Y-%m-%d %H:%M}</i> | panel {research['tarih'].nunique()} gün, {research['ticker'].nunique()} hisse\n━━━━━━━━━━━━━━━━━━━━\n"
    if wf.get("ok"):
        def line(tag, m):
            return (f"{tag}: N={m['n']} | WR %{m['win_rate']:.1f} (LCB %{m['wilson_lcb']:.1f}) | "
                    f"net %{m['avg_return']:+.2f} | PF {m['profit_factor']:.2f} | P10 %{m['p10']:+.1f} | t={m['cohort_t']:.1f}\n")
        rep += f"🧪 Walk-forward: {len(wf['folds'])} dilim, embargo {C.WF_EMBARGO_DAYS}g, T+{C.HORIZON} net\n"
        rep += line("🟢 AKTİF", wf["active"]) + line("🟡 ADAY", wf["candidate"]) + line("⚪ ŞABLON", wf["template"])
        rep += "📐 OOS IC: " + " | ".join(f"{k} {v['ic']:+.3f} (t {v['t']:.1f})" for k, v in wf["ic"].items()) + "\n"
        rep += f"🎚️ {meta.get('winrate_optimizer_status', '')}\n"
    rep += f"⚖️ <b>Karar:</b> {'TERFİ' if decision else ('ROLLBACK' if rollback else 'SHADOW/BEKLE')} — {note}\n"
    if live_m.get("n"):
        rep += f"📒 Canlı defter: N={live_m['n']} | WR %{live_m['win_rate']:.1f} | net %{live_m['avg_return']:+.2f} | PF {live_m['profit_factor']:.2f}\n"
    g = state.get("autonomy_guard", {})
    rep += f"🛡️ Otonomi: {g.get('mode')} ({g.get('reason')})\n"
    rep += f"⏱️ {(datetime.now() - t0).seconds}s"
    send_telegram_audit(rep)
    print("Denetim tamamlandı.")


# Geriye uyumluluk
run_evening_audit = run_audit

if __name__ == "__main__":
    run_audit()
