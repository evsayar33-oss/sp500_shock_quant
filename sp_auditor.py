"""S&P 500 Adaptive Meta-Engine — öğrenme ve denetim (kendini iyileştirme döngüsü) v2.2.

Her çalıştırmada:
 1) 3 yıllık panelden özellik + NET etiket + grup akışı + rejim (kesitsel+makro) üretilir.
 2) Embargo'lu walk-forward: aday ÖĞRENME PROSEDÜRÜ (ağırlık, işaret, eşik, yön kapısı) vs aktif profil.
 3) Terfi yalnızca OOS kanıtla: Wilson LCB / net ortalama / PF korumaları + t >= PROMOTION_MIN_T.
 4) Meta-etiket (kazanma olasılığı filtresi): OOS'ta win-rate'i yükseltip neti bozmuyorsa AÇILIR.
 5) Win-rate optimizer: global eşik ofseti (embargo'lu, günlük top-K sınırlı, ofset çift sayımı düzeltildi).
 6) Çıkış stratejisi karşılaştırması (zaman vs kâr-al) raporlanır.
 7) Canlı defter bozulursa stabil profile geri dönüş.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import health
import report
from autonomy_guard import evaluate_autonomy_guard
from price_history import load_earnings, load_macro, load_panel
from features import attach_regime, build_features
from meta_label import fit_meta
from regime import compute_macro_frame
from sp_engine import score_frame
from sp_learner import (_objective, active_profiles, ensure_meta_state, exit_compare, learn_profiles, live_rollback_needed,
                           load_ai_state, load_signal_history, meta_decision, promotion_decision, save_ai_state,
                           trade_metrics, walk_forward)
from win_rate_optimizer import optimize_win_rate, summary as winrate_summary

PROJECT_KEY = "sp500_shock"


def _slim(p):
    return {k: p.get(k) for k in ("weights", "signs", "sign_stats", "gate", "min_score", "regime", "version",
                                  "learned_edges", "regime_days", "threshold_source", "train_metrics") if k in p}


def build_research():
    panel = load_panel()
    if panel.empty:
        return pd.DataFrame(), panel
    feat = build_features(panel, with_labels=True, earnings=load_earnings())
    research = attach_regime(feat, compute_macro_frame(load_macro()))
    return research, panel


def run_audit():
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] {report.TITLE} öğrenme denetimi başlıyor...")
    state = load_ai_state()
    meta = ensure_meta_state(state)
    research, panel = build_research()
    if research.empty:
        report.send(f"⚠️ {report.TITLE}: geçmiş panel yok; denetim atlandı.")
        save_ai_state(state)
        return

    active = active_profiles(state)
    offset = float(np.clip(float(state.get("win_rate_optimizer", {}).get("active_threshold", 75.0)) - 75.0, -6.0, 10.0))
    wf = walk_forward(research, active, offset=offset)
    ctx = {"day": t0, "days": int(research["tarih"].nunique()), "tickers": int(research["ticker"].nunique()),
           "ok": bool(wf.get("ok")), "offset": offset}

    # Güçlü yeniden eğitim: canlı tetikleyici (health.retrain_triggers), sıfırlama ya da haftalık tam yenileme
    rt = state.setdefault("retrain", {"pending": False, "reasons": [], "history": []})
    strong = bool(rt.get("pending")) or os.environ.get("FULL_RETRAIN") == "1"
    train_mode, window = "Tam geçmiş", None
    if strong and wf.get("ok"):
        window = int(getattr(C, "ROLLING_TRAIN_DAYS", 375))
        wf_r = walk_forward(research, active, offset=offset, window=window)
        if wf_r.get("ok") and _objective(wf_r["candidate"]) > _objective(wf["candidate"]):
            for k in ("candidate", "candidate_trades", "pool", "folds", "ic"):
                wf[k] = wf_r[k]
            train_mode = f"Yakın dönem ({window} gün)"
        else:
            window = None
    ctx["retrain"] = {"strong": strong, "mode": train_mode,
                      "reasons": rt.get("reasons", []) or (["HAFTALIK"] if os.environ.get("FULL_RETRAIN") == "1" else [])}
    decision, note, status = False, "", "KEPT"

    if not wf.get("ok"):
        note = wf.get("reason", "WARMUP")
        ctx["note"] = note
    else:
        decision, note = promotion_decision(wf["active"], wf["candidate"])
        lab = research[research["net_ret"].notna()]
        if window:
            keep_days = sorted(lab["tarih"].unique())[-window:]
            lab = lab[lab["tarih"].isin(keep_days)]
        final = learn_profiles(lab)  # nihai aday: seçilen eğitim modu ile tüm uygun etiketli veri
        if decision:
            meta["stable_profiles"] = {r: _slim(p) for r, p in active.items()}
            meta["regime_profiles"] = {r: {**_slim(p), "promoted_at": t0.isoformat(timespec="seconds")} for r, p in final.items()}
            meta["promotion_count"] = int(meta.get("promotion_count", 0)) + 1
            meta["shadow"] = {}
            status = "PROMOTED"
        else:
            meta["shadow"] = {"profiles": {r: _slim(p) for r, p in final.items()},
                              "created_at": t0.isoformat(timespec="seconds"), "decision": note}
        live_profiles = active_profiles(state)

        # Meta-etiket kararı (OOS kanıt) + canlı profil ile nihai model
        # Meta, aktif skorlama üzerinde doğrulandı; bugün yeni profil terfi ettiyse bir sonraki denetimde
        # yeni aktif üzerinde yeniden doğrulanana kadar kapalı tutulur.
        meta_on, meta_note = meta_decision(wf["active"], wf["meta"], wf["folds"])
        if decision and meta_on:
            meta_on, meta_note = False, "Yeni model terfi etti; olasılık filtresi bir sonraki denetimde yeniden doğrulanacak"
        model = None
        if meta_on:
            model = fit_meta(score_frame(research[research["net_ret"].notna()], live_profiles, threshold_offset=offset),
                             trade_metrics)
            meta_on = model is not None
        state["meta_label"] = {"enabled": bool(meta_on), "note": meta_note, "oos": wf["meta"],
                               "base_oos": wf["active"], "model": model, "updated": t0.isoformat(timespec="seconds")}

        # Win-rate optimizer (global ofset). Havuz ofsetten arındırılmış -> current = 75 + offset tutarlı.
        pool = wf["pool"].rename(columns={"net_ret": "realized_5d"})
        state = optimize_win_rate(state, pool, current_threshold=75.0 + offset)
        meta["winrate_optimizer_status"] = winrate_summary(state)

        live_trades = wf["meta_trades"] if meta_on else (wf["candidate_trades"] if decision else wf["active_trades"])
        exits = exit_compare(live_trades)
        live_sc = wf["meta"] if meta_on else (wf["candidate"] if decision else wf["active"])
        state["backtest_summary"] = live_sc
        rep = {"generated": t0.isoformat(timespec="seconds"), "days": wf["days"], "horizon": C.HORIZON,
               "candidate_oos": wf["candidate"], "active_oos": wf["active"], "template_oos": wf["template"],
               "meta_oos": wf["meta"], "meta_enabled": bool(meta_on), "meta_note": meta_note,
               "exit_variants": {k: v for k, v in exits.items() if not k.startswith("_")}, "exit_rates": exits.get("_rates", {}),
               "exit_rule": {k: getattr(C, k, None) for k in ("EXIT_STOP_ATR", "EXIT_TP1_ATR", "EXIT_TP1_FRAC", "EXIT_TP2_ATR")},
               "train_mode": train_mode, "retrain_reasons": ctx["retrain"]["reasons"],
               "ic_oos": wf["ic"], "folds": wf["folds"], "decision": note, "promoted": decision}
        os.makedirs(C.DATA_DIR, exist_ok=True)
        with open(C.BACKTEST_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
        keep = ["tarih", "ticker", "shock_score", "effective_min_score", "regime_label", "macro_label",
                "net_ret", "gross_ret", "cost_rt", "gap_next", "fold"]
        tr = live_trades
        if not tr.empty:
            tr[[c for c in keep if c in tr.columns]].to_csv(C.OOS_TRADES_FILE, index=False,
                                                            compression={"method": "gzip", "mtime": 0})
        reg_now = state.get("last_scan", {}).get("regime", {}).get("label", "NORMAL")
        ctx.update({"active": wf["active"], "candidate": wf["candidate"], "meta": wf["meta"], "meta_on": meta_on,
                    "exits": exits, "drivers": report.drivers_from_profile(live_profiles.get(reg_now, {})),
                    "gate": (live_profiles.get(reg_now) or {}).get("gate", "UP")})

    # Canlı rollback
    from main import load_ledger
    rb, live_m = live_rollback_needed(load_ledger())
    if rb and meta.get("stable_profiles"):
        meta["regime_profiles"] = meta["stable_profiles"]
        meta["rollback_count"] = int(meta.get("rollback_count", 0)) + 1
        status, note = "ROLLBACK", "Canlı defter güvenlik sınırını bozdu"
    state["live_metrics"] = live_m

    sig = load_signal_history()
    perf = None
    if not sig.empty and "realized_5d" in sig.columns and "label_version" in sig.columns:
        perf = pd.to_numeric(sig.loc[pd.to_numeric(sig["label_version"], errors="coerce") == 2, "realized_5d"],
                             errors="coerce").dropna()
    last_reg = state.get("last_scan", {}).get("regime", {})
    g = evaluate_autonomy_guard(state, features=None, regime={"label": last_reg.get("label", "NORMAL")},
                                regime_confidence=float(last_reg.get("confidence", 0.5) or 0.5), performance_returns=perf,
                                data_quality_score=100.0, row_count=int(research["ticker"].nunique()), min_rows=100,
                                project=PROJECT_KEY, stress_test_due=False)

    meta["last_decision"] = {"promoted": decision, "rollback": status == "ROLLBACK", "note": note,
                             "timestamp": t0.isoformat(timespec="seconds"), "train_mode": train_mode}
    if strong:
        rt["history"] = (rt.get("history", []) + [{"ts": t0.isoformat(timespec="seconds"), "done": train_mode,
                                                    "reasons": ctx["retrain"]["reasons"], "promoted": decision}])[-30:]
        rt["pending"], rt["reasons"] = False, []
    health.heartbeat(state, "audit")
    if os.environ.get("FULL_RETRAIN") == "1":
        health.heartbeat(state, "weekly")
    save_ai_state(state)

    ctx.update({"status": status, "note": note, "guard": g.get("mode"), "live": live_m,
                "offset": float(state.get("win_rate_optimizer", {}).get("active_threshold", 75.0)) - 75.0})
    report.send(report.audit_message(ctx))
    print(f"Denetim tamamlandı ({(datetime.now() - t0).seconds}s).")


run_evening_audit = run_audit

if __name__ == "__main__":
    run_audit()
