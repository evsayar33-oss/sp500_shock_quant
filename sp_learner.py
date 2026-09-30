"""Öğrenme katmanı v2 — purged/embargo'lu walk-forward, rejim profilleri, canlı etiketleme.

Düzeltilen mantık hataları
--------------------------
* Öğrenme ve doğrulama AYNI hedefi kullanır: T+1 açılış -> T+HORIZON kapanış NET getiri.
* Eşik yalnızca EĞİTİM verisinde seçilir; test diliminde asla optimize edilmez.
* Eğitim sonu ile test başı arasında WF_EMBARGO_DAYS boşluk vardır (örtüşen etiket sızıntısı yok).
* IC günlük kesitsel Spearman olarak hesaplanır ve ortalanır (havuzlanmış korelasyon yanlılığı yok).
* Anlamlılık, örtüşmeyen kohortlar (her HORIZON günde bir) üzerinden t-istatistiği ile raporlanır.
* Canlı sinyal/defter etiketleri paneldeki KESİN tarihlerden hesaplanır; tarama kaçsa da T+5 kaymaz.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
from meta_label import fit_meta, predict as meta_predict, select_with_meta  # noqa: F401
from sp_engine import (DEFAULT_META_WEIGHTS, FAMILY_COLS, REGIME_META_TEMPLATES, REGIME_MIN_SCORES,
                          default_profile, default_profiles, normalize_weights, score_frame)

AI_STATE_FILE = C.AI_STATE_FILE
SIGNAL_LOG_FILE = C.SIGNAL_LOG_FILE
GECMIS_DOSYA = C.GECMIS_DOSYA
LEDGER_FILE = C.LEDGER_FILE
META_VERSION = 2
Z_90 = 1.2815515655446004


# ------------------------------------------------------------------
# Durum dosyası
# ------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def load_ai_state():
    if not os.path.exists(AI_STATE_FILE):
        return {}
    try:
        with open(AI_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def save_ai_state(state):
    tmp = AI_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, AI_STATE_FILE)


def ensure_meta_state(state):
    meta = state.setdefault("meta_engine", {})
    if int(_safe_float(meta.get("version"), 1)) < META_VERSION:
        # v1 profilleri farklı hedef/özelliklerle öğrenildi; arşivlenir, v2 şablonla başlar.
        arch = state.setdefault("archive_v1", {})
        arch["meta_engine"] = dict(meta)
        for k in ("win_rate_optimizer", "thresholds", "weights", "resilience_weight", "legacy_audit"):
            if k in state:
                arch[k] = state.pop(k)
        meta.clear()
    meta["version"] = META_VERSION
    meta.setdefault("regime_profiles", {})
    meta.setdefault("stable_profiles", {})
    meta.setdefault("shadow", {})
    meta.setdefault("promotion_count", 0)
    meta.setdefault("rollback_count", 0)
    return meta


def active_profiles(state):
    meta = ensure_meta_state(state)
    profs = default_profiles()
    for r, p in (meta.get("regime_profiles") or {}).items():
        if isinstance(p, dict) and p.get("weights"):
            profs[r] = p
    return profs


# ------------------------------------------------------------------
# Metrikler
# ------------------------------------------------------------------
def wilson_lower_bound(wins, n, z=Z_90):
    if n <= 0:
        return 0.0
    p = wins / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return float((centre - margin) / den)


def trade_metrics(trades: pd.DataFrame, ret_col="net_ret") -> dict:
    empty = {"n": 0, "wins": 0, "win_rate": 0.0, "wilson_lcb": 0.0, "profit_factor": 0.0, "avg_return": 0.0,
             "median": 0.0, "p10": 0.0, "cohort_t": 0.0, "n_cohorts": 0, "days": 0}
    if trades is None or trades.empty or ret_col not in trades.columns:
        return empty
    t = trades.dropna(subset=[ret_col])
    ret = t[ret_col].astype(float)
    if ret.empty:
        return empty
    wins = int((ret > 0).sum())
    n = int(len(ret))
    gains, losses = float(ret[ret > 0].sum()), float(abs(ret[ret < 0].sum()))
    pf = gains / losses if losses > 0 else (5.0 if gains > 0 else 0.0)
    # Günlük eşit ağırlıklı kohort getirileri; HORIZON gün örtüşmesi için etkin örneklem n/HORIZON.
    # (v2.1'deki "her 5. işlem günü" alt örneklemesi başlangıç gününe bağlıydı ve t'yi rastgele saptırıyordu.)
    daily = t.groupby("tarih")[ret_col].mean().sort_index()
    cohorts = daily
    cohort_t = 0.0
    if len(daily) >= 10 and daily.std(ddof=1) > 0:
        cohort_t = float(daily.mean() / daily.std(ddof=1) * math.sqrt(len(daily) / C.HORIZON))
    return {"n": n, "wins": wins, "win_rate": round(wins / n * 100.0, 2),
            "wilson_lcb": round(wilson_lower_bound(wins, n) * 100.0, 2), "profit_factor": round(pf, 3),
            "avg_return": round(float(ret.mean()), 3), "median": round(float(ret.median()), 3),
            "p10": round(float(ret.quantile(0.10)), 3), "cohort_t": round(cohort_t, 2),
            "n_cohorts": int(len(cohorts) // C.HORIZON), "days": int(daily.shape[0])}


def select_trades(scored: pd.DataFrame, top_k=C.TOP_K_PER_DAY, threshold=None) -> pd.DataFrame:
    """Canlı ile aynı seçim: uygun + skor >= efektif eşik, her gün en iyi top_k."""
    if scored is None or scored.empty:
        return pd.DataFrame()
    thr = scored["effective_min_score"] if threshold is None else threshold
    sel = scored[scored["eligible"] & (scored["shock_score"] >= thr)]
    if sel.empty:
        return sel
    sel = sel.sort_values(["tarih", "shock_score"], ascending=[True, False])
    return sel.groupby("tarih", sort=False).head(top_k)


# ------------------------------------------------------------------
# Öğrenme
# ------------------------------------------------------------------
def daily_rank_ic(df: pd.DataFrame, col: str, target="net_ret") -> tuple[float, float, int]:
    """Günlük kesitsel Spearman IC ortalaması, t-istatistiği, gün sayısı."""
    d = df[["tarih", col, target]].dropna()
    if d.empty:
        return 0.0, 0.0, 0
    g = d.groupby("tarih")
    rx = g[col].rank()
    ry = g[target].rank()
    dx = rx - rx.groupby(d["tarih"]).transform("mean")
    dy = ry - ry.groupby(d["tarih"]).transform("mean")
    num = (dx * dy).groupby(d["tarih"]).sum()
    den = np.sqrt((dx ** 2).groupby(d["tarih"]).sum() * (dy ** 2).groupby(d["tarih"]).sum())
    cnt = d.groupby("tarih").size()
    ic = (num / den.replace(0, np.nan))[cnt >= 8].dropna()
    if len(ic) < 5:
        return 0.0, 0.0, int(len(ic))
    # Etiketler HORIZON gün örtüştüğü için etkin örneklem n/HORIZON kabul edilir
    sd = ic.std(ddof=1)
    t = float(ic.mean() / sd * math.sqrt(len(ic) / C.HORIZON)) if sd > 0 else 0.0
    return float(ic.mean()), t, int(len(ic))


SIGN_FLIP_T = -2.0      # bir ailenin işareti yalnızca eğitimde t <= -2 ise çevrilir (muhafazakâr)


def learn_signs(train: pd.DataFrame) -> tuple[dict, dict]:
    """Aile işaretleri: tüm eğitim havuzunda (işlem yapılan popülasyon) günlük rank-IC ve t."""
    pool = train[train["current_positive"] & ~train["is_illiquid"] & train["net_ret"].notna()]
    signs, stats = {}, {}
    for fam, col in FAMILY_COLS.items():
        ic, t, n = daily_rank_ic(pool, col)
        signs[fam] = -1 if t <= SIGN_FLIP_T else 1
        stats[fam] = {"ic": round(ic, 4), "t": round(t, 2), "days": n}
    return signs, stats


def _learn_weights(train: pd.DataFrame, regime: str, signs: dict | None = None) -> tuple[dict, dict, int]:
    template = REGIME_META_TEMPLATES.get(regime, DEFAULT_META_WEIGHTS)
    pool = train[(train["regime_label"] == regime)]
    n_days = pool["tarih"].nunique()
    if n_days < 40:
        pool = train
        n_days_used = 0
    else:
        n_days_used = n_days
    pool = pool[pool["current_positive"] & ~pool["is_illiquid"]]  # işlem yapılan popülasyonda IC
    signs = signs or {k: 1 for k in FAMILY_COLS}
    edges = {}
    for fam, col in FAMILY_COLS.items():
        ic, t, _ = daily_rank_ic(pool, col)
        ic, t = ic * signs.get(fam, 1), t * signs.get(fam, 1)   # işarete hizalanmış kenar
        # Yalnızca (hizalanmış) pozitif kenar ağırlık kazanır; aksi halde küçük taban
        edges[fam] = max(ic, 0.0) * (1.0 if t >= 1.0 else 0.5) + 0.005
    total = sum(edges.values())
    learned = {k: v / total for k, v in edges.items()}
    blend = float(np.clip((n_days_used or pool["tarih"].nunique() * 0.5) / 500.0, 0.10, 0.60))
    w = normalize_weights({k: (1 - blend) * template[k] + blend * learned[k] for k in template}, template)
    return {k: round(v, 4) for k, v in w.items()}, {k: round(v, 4) for k, v in edges.items()}, int(n_days_used)


def _pick_threshold(scored_regime: pd.DataFrame, floor: float) -> tuple[float, dict | None]:
    """Eğitimde: Wilson LCB (win-rate) maksimizasyonu; kısıt: net ort > 0, PF >= 1.05, yeterli n."""
    best = None
    base = scored_regime[scored_regime["eligible"]]
    add = base["effective_min_score"] - base["_min_used"]  # makro + ofset primi
    for th in C.THRESHOLD_GRID:
        if th < floor:
            continue
        sel = base[base["shock_score"] >= th + add]
        if sel.empty:
            continue
        sel = sel.sort_values(["tarih", "shock_score"], ascending=[True, False]).groupby("tarih").head(C.TOP_K_PER_DAY)
        m = trade_metrics(sel)
        if m["n"] < C.MIN_TRADES_FOR_THRESHOLD or m["avg_return"] <= 0 or m["profit_factor"] < 1.05:
            continue
        key = (m["wilson_lcb"], m["avg_return"])
        if best is None or key > best[0]:
            best = (key, th, m)
    return (best[1], best[2]) if best else (None, None)


GATE_MODES = ("UP", "ANY")


def _objective(m: dict) -> float:
    """Eğitim hedefi: win-rate güven alt sınırı + net getiri (win-rate'i maliyetsiz şişirmeyi engeller)."""
    if m["n"] < C.MIN_TRADES_FOR_THRESHOLD:
        return -1e9
    return m["wilson_lcb"] + 5.0 * m["avg_return"]


def learn_profiles(train: pd.DataFrame) -> dict:
    """Her rejim için ağırlık + eşik + işaret öğrenir; yön kapısını (UP/ANY) eğitimde seçer."""
    best = None
    for gate in GATE_MODES:
        profs = _learn_profiles_gate(train, gate)
        m = trade_metrics(select_trades(score_frame(train[train["net_ret"].notna()], profs)))
        obj = _objective(m)
        if best is None or obj > best[0]:
            best = (obj, profs, m)
    profs = best[1]
    for p in profs.values():
        p["gate_train_metrics"] = best[2]
    return profs


def _learn_profiles_gate(train: pd.DataFrame, gate: str) -> dict:
    train = train[train["net_ret"].notna()]
    signs, sign_stats = learn_signs(train)
    profiles = {}
    for reg in C.REGIMES:
        w, edges, n_days = _learn_weights(train, reg, signs)
        profiles[reg] = {"weights": w, "signs": dict(signs), "gate": gate, "min_score": REGIME_MIN_SCORES[reg], "regime": reg,
                         "version": META_VERSION, "learned_edges": edges, "regime_days": n_days,
                         "sign_stats": sign_stats}
    scored = score_frame(train, profiles)
    scored["_min_used"] = scored["regime_label"].map(lambda r: profiles[r]["min_score"])
    global_th, _ = _pick_threshold(scored, floor=min(REGIME_MIN_SCORES.values()) - 6)
    for reg in C.REGIMES:
        floor = REGIME_MIN_SCORES[reg] - 6.0
        part = scored[scored["regime_label"] == reg]
        th, m = (None, None)
        if part["tarih"].nunique() >= 40:
            th, m = _pick_threshold(part, floor=floor)
        if th is None:
            th = max(global_th, floor) if global_th is not None else REGIME_MIN_SCORES[reg]
            profiles[reg]["threshold_source"] = "GLOBAL" if global_th is not None else "TEMPLATE"
        else:
            profiles[reg]["threshold_source"] = "REGIME"
            profiles[reg]["train_metrics"] = m
        profiles[reg]["min_score"] = float(th)
    return profiles


# ------------------------------------------------------------------
# Walk-forward
# ------------------------------------------------------------------
def walk_forward(research: pd.DataFrame, active: dict, offset: float = 0.0, window: int | None = None) -> dict:
    """Genişleyen pencere, embargo'lu walk-forward. Aday ÖĞRENME PROSEDÜRÜ ile aktif profili
    aynı test dilimlerinde karşılaştırır."""
    lab = research[research["net_ret"].notna()]
    dates = np.array(sorted(lab["tarih"].unique()))
    if len(dates) < C.WF_MIN_TRAIN_DAYS + C.WF_EMBARGO_DAYS + 20:
        return {"ok": False, "reason": f"WARMUP ({len(dates)} etiketli gün)", "days": int(len(dates))}

    folds, cand_tr, act_tr, tmpl_tr, pool, meta_tr = [], [], [], [], [], []
    start = C.WF_MIN_TRAIN_DAYS + C.WF_EMBARGO_DAYS
    tmpl = default_profiles()
    while start < len(dates):
        test_dates = dates[start:start + C.WF_TEST_DAYS]
        train_dates = dates[:start - C.WF_EMBARGO_DAYS]
        if window:  # yakın dönem adayı: yalnızca son `window` etiketli günle öğren
            train_dates = train_dates[-int(window):]
        train = lab[lab["tarih"].isin(train_dates)]
        test = research[research["tarih"].isin(test_dates)]
        cand = learn_profiles(train)
        sc_c = score_frame(test, cand, threshold_offset=offset)
        sc_a = score_frame(test, active, threshold_offset=offset)
        sc_t = score_frame(test, tmpl, threshold_offset=offset)
        tc, ta, tt = select_trades(sc_c), select_trades(sc_a), select_trades(sc_t)
        # Meta-etiket: CANLIDA KULLANILAN (aktif) skorlama üzerinde eğitimde kurulur, testte uygulanır.
        # (Aday profil aynı eğitim diliminde öğrenildiği için onun skorları örneklem içidir ve meta modeli yanıltır.)
        model = fit_meta(score_frame(train, active, threshold_offset=offset), trade_metrics)
        tm = select_with_meta(sc_a, model) if model else ta
        meta_tr.append(tm.assign(fold=len(folds)))
        cand_tr.append(tc.assign(fold=len(folds)))
        act_tr.append(ta.assign(fold=len(folds)))
        tmpl_tr.append(tt.assign(fold=len(folds)))
        # win-rate optimizer için eşik-altı dahil aday havuzu. Marj, MEVCUT OFSET HARİÇ eşiğe göre
        # normalize edilir (v2.1 hatası: ofset havuza da gömülüyordu, her çalıştırmada üst üste biniyordu).
        p = sc_c[sc_c["eligible"] & (sc_c["shock_score"] >= sc_c["effective_min_score"] - 10.0)].copy()
        p = p.sort_values(["tarih", "shock_score"], ascending=[True, False]).groupby("tarih").head(15)
        p["signal_score"] = p["shock_score"] - (p["effective_min_score"] - offset) + 75.0
        pool.append(p[["tarih", "ticker", "signal_score", "net_ret"]])
        folds.append({"train_end": str(pd.Timestamp(train_dates[-1]).date()),
                      "test_start": str(pd.Timestamp(test_dates[0]).date()),
                      "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                      "candidate": trade_metrics(tc), "active": trade_metrics(ta), "meta": trade_metrics(tm),
                      "meta_on": bool(model), "gate": next(iter(cand.values())).get("gate", "UP")})
        start += C.WF_TEST_DAYS

    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    cand_all, act_all, tmpl_all, meta_all = cat(cand_tr), cat(act_tr), cat(tmpl_tr), cat(meta_tr)
    oos_idx = research["tarih"].isin(cand_all["tarih"].unique()) if not cand_all.empty else research["tarih"].isin([])
    ic_report = {}
    oos = research[oos_idx & research["current_positive"] & ~research["is_illiquid"]]
    for fam, col in FAMILY_COLS.items():
        ic, t, n = daily_rank_ic(oos, col)
        ic_report[fam] = {"ic": round(ic, 4), "t": round(t, 2), "days": n}
    return {"ok": True, "folds": folds, "candidate": trade_metrics(cand_all), "active": trade_metrics(act_all),
            "template": trade_metrics(tmpl_all), "meta": trade_metrics(meta_all), "ic": ic_report,
            "candidate_trades": cand_all, "meta_trades": meta_all, "active_trades": act_all,
            "pool": cat(pool), "days": int(len(dates))}


PROMOTION_MIN_T = getattr(C, "PROMOTION_MIN_T", 1.0)
META_MIN_T = getattr(C, "META_MIN_T", 0.0)   # yön şartı (t>0); asıl kanıt LCB + dilim tutarlılığı
META_MIN_FOLD_SHARE = getattr(C, "META_MIN_FOLD_SHARE", 0.6)


def meta_decision(base_m: dict, meta_m: dict, folds: list | None = None, was_on: bool = False) -> tuple[bool, str]:
    """Kazanma olasılığı filtresinin açık/kapalı kararı — HİSTEREZİSLİ.

    Açmak için (hepsi): dilimlerin >= %60'ında net üstünlük, toplam net kazanç >= +0.20 puan,
      PF >= aktif PF, kazanma oranı >= +1 puan, yeterli işlem.
    Açıkken kapatmak için (herhangi biri): dilimlerin < %50'sinde üstünlük, net kazanç < 0, PF < aktif PF,
      kazanma oranı < −1 puan.
    Neden: filtre aktif modelden ~5 kat az işlem yaptığı için güven aralığı doğal olarak geniştir; LCB
    karşılaştırması küçük veri değişikliklerinde kararı gün içinde çevirebiliyordu (S&P, 30 Eyl: 11 dk arayla
    açık→kapalı). Tutarlılık + ekonomik fark + histerezis kararlı ve savunulabilir bir kuraldır."""
    if meta_m["n"] < C.PROMOTION_MIN_OOS_TRADES:
        return False, f"meta OOS örneklem yetersiz (n={meta_m['n']})"
    wr = meta_m["win_rate"] - base_m["win_rate"]
    avg = meta_m["avg_return"] - base_m["avg_return"]
    pf_ok = meta_m["profit_factor"] >= base_m["profit_factor"]
    fl = [f for f in (folds or []) if (f.get("meta") or {}).get("n", 0) >= 10 and (f.get("active") or {}).get("n", 0) >= 10]
    wins = sum(1 for f in fl if f["meta"]["avg_return"] > f["active"]["avg_return"])
    share = wins / len(fl) if fl else 0.0
    if was_on:
        ok = share >= 0.5 and avg >= 0 and pf_ok and wr >= -1.0
        mode = "korunuyor" if ok else "kapatıldı"
    else:
        ok = share >= META_MIN_FOLD_SHARE and avg >= 0.20 and pf_ok and wr >= 1.0 and meta_m["cohort_t"] >= META_MIN_T
        mode = "açıldı" if ok else "kapalı"
    return bool(ok), (f"filtre {mode} | kazanma {wr:+.1f}pp | net {avg:+.2f} | PF {meta_m['profit_factor']:.2f} vs "
                      f"{base_m['profit_factor']:.2f} | dilim tutarlılığı {wins}/{len(fl)}")


def promotion_decision(active_m: dict, cand_m: dict) -> tuple[bool, str]:
    if cand_m["n"] < C.PROMOTION_MIN_OOS_TRADES:
        return False, f"OOS örneklem yetersiz (n={cand_m['n']})"
    lcb_lift = cand_m["wilson_lcb"] - active_m["wilson_lcb"]
    avg_lift = cand_m["avg_return"] - active_m["avg_return"]
    if cand_m["avg_return"] <= 0 or cand_m["profit_factor"] < 1.05:
        return False, f"Aday mutlak tabanı geçemedi (ort {cand_m['avg_return']:+.2f}, PF {cand_m['profit_factor']:.2f})"
    if active_m["n"] >= C.PROMOTION_MIN_OOS_TRADES and cand_m["profit_factor"] < 0.95 * active_m["profit_factor"]:
        return False, f"PF bozulması ({cand_m['profit_factor']:.2f} vs {active_m['profit_factor']:.2f})"
    improve = (lcb_lift >= C.PROMOTION_MIN_LCB_LIFT and avg_lift >= -0.05) or \
              (avg_lift >= C.PROMOTION_MIN_AVG_LIFT and lcb_lift >= -1.0)
    note = f"LCB fark {lcb_lift:+.2f}pp | Net ort fark {avg_lift:+.2f} | t={cand_m['cohort_t']:.2f}"
    active_broken = active_m["n"] >= C.PROMOTION_MIN_OOS_TRADES and (active_m["avg_return"] <= 0 or active_m["profit_factor"] < 1.0)
    if active_broken and lcb_lift > 0 and avg_lift > 0 and cand_m["cohort_t"] > 0:
        # Aktif model mutlak tabanın altında (zarar ediyor): tabanı geçen ve her iki ölçüde daha iyi aday
        # güven eşiğini beklemeden terfi eder. Güven eşiği iyi modeli korumak içindir, kötü modelden kaçışı engellemez.
        return True, "Aktif model zarar ediyor; daha iyi aday devreye alındı | " + note
    if cand_m["cohort_t"] < PROMOTION_MIN_T:
        return False, f"İstatistiksel güven yetersiz (t={cand_m['cohort_t']:.2f} < {PROMOTION_MIN_T}) | " + note
    if active_m["n"] < C.PROMOTION_MIN_OOS_TRADES:
        return True, "Aktif profil OOS'ta yetersiz işlem üretti; aday tabanı geçti | " + note
    return bool(improve), note


def live_rollback_needed(ledger: pd.DataFrame) -> tuple[bool, dict]:
    """v3 defterde kapanmış canlı işlemler (çıkış kuralıyla net) bozulursa stabil profile dönülür."""
    if ledger is None or ledger.empty or "net_ret" not in ledger.columns:
        return False, {}
    lv = pd.to_numeric(ledger.get("label_version"), errors="coerce")
    done = ledger[(lv == 3) & (ledger.get("status") == "CLOSED")].copy()
    done["tarih"] = pd.to_datetime(done.get("date"), errors="coerce")
    done["net_ret"] = pd.to_numeric(done["net_ret"], errors="coerce")
    done = done.sort_values("tarih").tail(C.LIVE_ROLLBACK_MIN_TRADES * 2)
    m = trade_metrics(done)
    if m["n"] < C.LIVE_ROLLBACK_MIN_TRADES:
        return False, m
    return bool(m["profit_factor"] < 0.80 or (m["wilson_lcb"] < 35.0 and m["avg_return"] < -0.5)), m


# ------------------------------------------------------------------
# Canlı sinyal logu ve etiketleme (kesin tarihlerle)
# ------------------------------------------------------------------
def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNAL_LOG_FILE)
        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


LOG_COLS = ["tarih", "ticker", "close", "shock_score", "watch_score", "meta_score", "risk_adjusted_score",
            "effective_min_score", "z_vol", "z_range", "z_flow", "z_lambda", "cmf20", "resilience_score",
            "grp", "sector_score", "sec_cmf", "sec_ret20",
            "excess_return", "rel_1m_pct", "rel_3m_pct", "trend_persistence", "event_score", "flow_score",
            "activity_score", "liquidity_score", "non_price_score", "overnight_risk", "liq20", "volatility",
            "current_positive", "crash_resilient", "crash_survivor", "meta_regime", "meta_regime_confidence",
            "macro_label", "macro_stress", "meta_selection", "weight_pct"]


def log_shock_signals(top_df, regime_snapshot=None, shadow_df=None):
    if top_df is None or top_df.empty:
        return
    frames = [top_df.assign(model_variant="active", signal_score=top_df["shock_score"])]
    if shadow_df is not None and not shadow_df.empty:
        frames.append(shadow_df.assign(model_variant="shadow", signal_score=shadow_df["watch_score"]))
    sig = pd.concat(frames, ignore_index=True, sort=False)
    for col in LOG_COLS:
        if col not in sig.columns:
            sig[col] = np.nan
    sig = sig[LOG_COLS + ["model_variant", "signal_score"]].copy()
    sig["tarih"] = pd.to_datetime(sig["tarih"]).dt.normalize()
    rs = regime_snapshot or {}
    sig["regime_label"] = rs.get("label", "NORMAL")
    sig["regime_confidence"] = _safe_float(rs.get("confidence"), 0.35)
    sig["label_version"] = 2
    for c in ("entry_price", "realized_1d", "realized_3d", "realized_5d", "gross_5d", "cost_rt"):
        sig[c] = np.nan

    hist = load_signal_history()
    day = sig["tarih"].iloc[0]
    if not hist.empty:
        hist = hist[pd.to_datetime(hist["tarih"], errors="coerce").dt.normalize() != day]
        sig = pd.concat([hist, sig], ignore_index=True, sort=False)
    sig.to_csv(SIGNAL_LOG_FILE, index=False)


def _label_rows(df, date_col, ticker_col, panel, liq_col=None):
    """Her satır için panelden kesin T+1 açılış ve T+k kapanışları (k=1,3,HORIZON)."""
    if df.empty or panel is None or panel.empty:
        return {}
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(panel["tarih"].unique())))
    px = panel.set_index(["tarih", "ticker"])[["open", "close"]]
    res = {}
    for idx, row in df.iterrows():
        d = pd.Timestamp(row[date_col]).normalize() if pd.notna(row[date_col]) else None
        t = row[ticker_col]
        if d is None:
            continue
        pos = int(cal.searchsorted(d, side="right"))  # T+1 indeksi
        out = {}
        if pos < len(cal) and (cal[pos], t) in px.index:
            entry = float(px.loc[(cal[pos], t), "open"])
            if entry > 0:
                out["entry_date"] = pd.Timestamp(cal[pos]).strftime("%Y-%m-%d")
                out["entry_price"] = round(entry, 4)
                liq = _safe_float(row.get(liq_col), C.MIN_LIQ_TL) if liq_col else C.MIN_LIQ_TL
                out["cost_rt"] = round(float(C.round_trip_cost_pct(liq)), 3)
                for k in (1, 3, C.HORIZON):
                    j = pos + k - 1
                    if j < len(cal) and (cal[j], t) in px.index:
                        cl = float(px.loc[(cal[j], t), "close"])
                        out[f"close_d{k}"] = round(cl, 4)
                        out[f"gross_d{k}"] = round((cl / entry - 1.0) * 100.0, 3)
                        if k == C.HORIZON:
                            out["exit_date"] = pd.Timestamp(cal[j]).strftime("%Y-%m-%d")
        res[idx] = out
    return res


def update_realized_shock_returns(panel):
    """Sinyal logundaki v2 satırlarını panelden kesin tarihlerle etiketler."""
    hist = load_signal_history()
    if hist.empty or panel is None or panel.empty:
        return hist
    if "label_version" not in hist.columns:
        hist["label_version"] = 1
    for c in ("entry_price", "realized_1d", "realized_3d", "realized_5d", "gross_5d", "cost_rt"):
        if c not in hist.columns:
            hist[c] = np.nan
    todo = hist[(hist["label_version"] == 2) & hist["realized_5d"].isna()]
    labels = _label_rows(todo, "tarih", "ticker", panel, liq_col="liq20")
    for idx, lab in labels.items():
        if "entry_price" in lab:
            hist.at[idx, "entry_price"] = lab["entry_price"]
            hist.at[idx, "cost_rt"] = lab["cost_rt"]
        for k, col in ((1, "realized_1d"), (3, "realized_3d")):
            if f"gross_d{k}" in lab:
                hist.at[idx, col] = lab[f"gross_d{k}"]
        if f"gross_d{C.HORIZON}" in lab:
            hist.at[idx, "gross_5d"] = lab[f"gross_d{C.HORIZON}"]
            hist.at[idx, "realized_5d"] = round(lab[f"gross_d{C.HORIZON}"] - lab["cost_rt"], 3)
    hist.to_csv(SIGNAL_LOG_FILE, index=False)
    return hist


# ------------------------------------------------------------------
# Geriye uyumlu küçük yardımcılar
# ------------------------------------------------------------------
def build_runtime_meta_profile(regime_snapshot, state=None):
    state = state if isinstance(state, dict) else load_ai_state()
    regime = str((regime_snapshot or {}).get("label", "NORMAL")).upper()
    prof = active_profiles(state).get(regime, default_profile(regime))
    return {"version": META_VERSION, "regime": regime, "weights": prof.get("weights"),
            "min_score": prof.get("min_score"), "source": "REGIME_PROFILE_v2"}


# ------------------------------------------------------------------
# Çıkış kuralı karşılaştırması (rapor): aktif kural (stop/TP1/TP2) vs yalnız T+H süre çıkışı
# ------------------------------------------------------------------
def exit_compare(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}
    out = {"Aktif kural": trade_metrics(trades)}
    if "net_ret_time" in trades.columns:
        out[f"Yalnız T+{C.HORIZON}"] = trade_metrics(trades.assign(net_ret=trades["net_ret_time"]))
    rates = {}
    for k, col in (("tp1", "tp1_hit"), ("tp2_given_tp1", "tp2_hit"), ("stop", "stop_hit")):
        if col in trades.columns:
            v = pd.to_numeric(trades[col], errors="coerce")
            if k == "tp2_given_tp1":
                base = pd.to_numeric(trades["tp1_hit"], errors="coerce")
                rates[k] = round(float(v.sum() / max(base.sum(), 1) * 100), 1)
            else:
                rates[k] = round(float(v.mean() * 100), 1)
    out["_rates"] = rates
    return out


def exit_variants(trades, panel=None, ks=None):  # geriye uyum
    return exit_compare(trades)
