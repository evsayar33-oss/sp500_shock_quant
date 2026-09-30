"""v3 SIFIRLAMA (tek seferlik): eski backtest / canlı defter / sinyal kayıtlarını siler, temiz başlangıç yapar.

Silinen (hepsi git geçmişinde kalır, gerekirse geri alınabilir):
  * işlem defteri (v1/v2 satırları)          -> boş v3 defter
  * sinyal kaydı                              -> silinir
  * data/backtest_report.json, oos_trades     -> silinir (sıfırlama workflow'u hemen yenisini üretir)
  * data/live_*                               -> silinir
  * durum dosyasındaki öğrenilmiş profiller, meta filtre, eşik ofseti, backtest karnesi, canlı metrikler, arşiv
Korunan: fiyat paneli (data/ohlcv), makro, evren, bilanço tarihleri, otonomi koruması, günlük kesit geçmişi.
Sonra: denetim FULL_RETRAIN=1 ile çalışır -> yeni çıkış kuralıyla yepyeni walk-forward backtest.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import config as C
import live_book as LB

now = datetime.now().isoformat(timespec="seconds")
removed = []

# 1) Defter -> boş v3
import pandas as pd  # noqa: E402

pd.DataFrame(columns=LB.TRADE_COLS).to_csv(C.LEDGER_FILE, index=False)
removed.append(C.LEDGER_FILE + " (boş v3 defter yazıldı)")

# 2) Kayıt ve backtest dosyaları
for f in (C.SIGNAL_LOG_FILE, C.BACKTEST_REPORT_FILE, C.OOS_TRADES_FILE,
          os.path.join(C.DATA_DIR, "live_trades.csv"), os.path.join(C.DATA_DIR, "live_weekly.csv"),
          os.path.join(C.DATA_DIR, "live_summary.json"), os.path.join(C.DATA_DIR, "health.json")):
    if os.path.exists(f):
        os.remove(f)
        removed.append(f)

# 3) Durum dosyası: öğrenilmiş her şey sıfırlanır, altyapı durumu korunur
state = {}
if os.path.exists(C.AI_STATE_FILE):
    try:
        state = json.load(open(C.AI_STATE_FILE, encoding="utf-8"))
    except Exception:
        state = {}
keep = {k: state[k] for k in ("autonomy_guard", "autonomy", "heartbeat", "last_scan", "regime_hist") if k in state}
g = keep.get("autonomy_guard")
if isinstance(g, dict):  # performans kayması eski etiketlerle ölçülmüştü -> sıfırla
    g.update({"performance_drift": 0.0, "mode": "NORMAL", "watch_streak": 0, "safe_streak": 0,
              "recovery_streak": 0, "reason": "RESET_V3", "performance_detail": {},
              "feature_baseline": {}, "drift_score": 0.0})  # v3 özellikleriyle yeniden öğrenilir
new_state = {
    **keep,
    "meta_engine": {"version": 2, "regime_profiles": {}, "stable_profiles": {}, "shadow": {},
                    "promotion_count": 0, "rollback_count": 0},
    "meta_label": {"enabled": False, "note": "v3 sıfırlama; ilk denetimde yeniden doğrulanacak"},
    "retrain": {"pending": True, "reasons": ["SIFIRLAMA_V3"], "history": []},
    "reset": {"ts": now, "version": 3, "removed": removed,
              "exit_rule": {k: getattr(C, k, None) for k in ("EXIT_STOP_ATR", "EXIT_TP1_ATR", "EXIT_TP1_FRAC", "EXIT_TP2_ATR")}},
    "status": "v3 sıfırlandı — yeni backtest bekleniyor",
}
tmp = C.AI_STATE_FILE + ".tmp"
json.dump(new_state, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
os.replace(tmp, C.AI_STATE_FILE)
os.makedirs(C.DATA_DIR, exist_ok=True)
json.dump(new_state["reset"], open(os.path.join(C.DATA_DIR, "reset_info.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("✅ v3 sıfırlama tamam. Silinen/yenilenen:")
for r in removed:
    print("  -", r)
