"""Project-local Win-Rate Optimizer V1.

Primary objective: improve out-of-sample win rate.
Safety constraints: minimum sample, Wilson lower bound, and (when returns exist)
profit factor / average-return deterioration limits.

The optimizer is deliberately narrow: it searches a small neighborhood around the
current score threshold and uses chronological walk-forward evaluation. It never
optimizes on the final live period and never replaces state on insufficient evidence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import math
import numpy as np
import pandas as pd


PROJECT = 'sp500_shock'
SCORE_COL = 'initial_score'
RETURN_COL = 'return_d5'
OUTCOME_COL = 'entry_status'
DATE_COL = 'date'
DEFAULT_THRESHOLD = 75.0
initial_score = "initial_score"
return_d5 = "return_d5"
entry_status = "entry_status"
date = "date"

MIN_TOTAL_SAMPLES = 60
MIN_TRAIN_SAMPLES = 40
MIN_TEST_SAMPLES = 20
MIN_TEST_WIN_LIFT = 0.02          # +2.0 percentage points
MIN_LCB_LIFT = 0.015              # +1.5 percentage points
MIN_PF_RATIO = 0.90               # PF cannot fall >10% vs active
MAX_AVG_RETURN_DROP = 0.25        # percentage-point tolerance
Z_90 = 1.2815515655446004


def _num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def wilson_lower_bound(wins: int, n: int, z: float = Z_90) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    den = 1.0 + (z * z / n)
    centre = p + (z * z / (2.0 * n))
    margin = z * math.sqrt((p * (1.0 - p) + (z * z / (4.0 * n))) / n)
    return float((centre - margin) / den)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if date not in out.columns or initial_score not in out.columns:
        return pd.DataFrame()
    out[date] = pd.to_datetime(out[date], errors="coerce").dt.normalize()
    out[initial_score] = pd.to_numeric(out[initial_score], errors="coerce")
    out = out.dropna(subset=[date, initial_score]).sort_values(date).copy()
    return out


def _wins_mask(df: pd.DataFrame) -> pd.Series:
    if return_d5 in df.columns:
        r = pd.to_numeric(df[return_d5], errors="coerce")
        return r.notna() & (r > 0.0)
    if entry_status in df.columns:
        s = df[entry_status].astype(str).str.upper()
        return s.str.startswith("WIN")
    return pd.Series(False, index=df.index)


def _resolved_mask(df: pd.DataFrame) -> pd.Series:
    if return_d5 in df.columns:
        return pd.to_numeric(df[return_d5], errors="coerce").notna()
    if entry_status in df.columns:
        s = df[entry_status].astype(str).str.upper()
        return s.isin({"WIN", "LOSS", "FAIL", "FAILED", "TIMEOUT_DEAD_INCUBATION", "FAIL_BASE_BREAKDOWN"}) | s.str.startswith("WIN")
    return pd.Series(False, index=df.index)


def _metrics(df: pd.DataFrame, threshold: float) -> Dict:
    if df.empty:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "wilson_lcb": 0.0, "pf": None, "avg_return": None}
    sel = df[pd.to_numeric(df[initial_score], errors="coerce") >= float(threshold)].copy()
    resolved = _resolved_mask(sel)
    sel = sel[resolved].copy()
    n = int(len(sel))
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "wilson_lcb": 0.0, "pf": None, "avg_return": None}
    wins = int(_wins_mask(sel).sum())
    wr = wins / n
    result = {
        "n": n,
        "wins": wins,
        "win_rate": round(wr * 100.0, 2),
        "wilson_lcb": round(wilson_lower_bound(wins, n) * 100.0, 2),
        "pf": None,
        "avg_return": None,
    }
    if return_d5 in sel.columns:
        ret = pd.to_numeric(sel[return_d5], errors="coerce").dropna()
        if not ret.empty:
            gains = float(ret[ret > 0].sum())
            losses = float(abs(ret[ret < 0].sum()))
            result["pf"] = round(gains / losses, 4) if losses > 0 else (5.0 if gains > 0 else 0.0)
            result["avg_return"] = round(float(ret.mean()), 4)
    return result


def _candidate_thresholds(current: float) -> list[float]:
    c = float(np.clip(current, 1.0, 99.0))
    vals = [c - 8, c - 5, c - 3, c, c + 3, c + 5, c + 8]
    return sorted({round(float(np.clip(v, 1.0, 99.0)), 1) for v in vals})


def _select_training_threshold(train: pd.DataFrame, current: float) -> Tuple[float, Dict]:
    candidates = []
    for th in _candidate_thresholds(current):
        m = _metrics(train, th)
        if m["n"] < MIN_TRAIN_SAMPLES:
            continue
        if m["pf"] is not None and m["pf"] < 0.90:
            continue
        candidates.append((th, m))
    if not candidates:
        return float(current), _metrics(train, current)
    # Primary objective = conservative win-rate estimate; then raw WR; then PF.
    candidates.sort(key=lambda x: (x[1]["wilson_lcb"], x[1]["win_rate"], x[1]["pf"] or -999), reverse=True)
    return candidates[0]


def _walk_forward(df: pd.DataFrame, current_threshold: float) -> Dict:
    dates = sorted(df[date].dropna().unique())
    if len(df) < MIN_TOTAL_SAMPLES or len(dates) < 6:
        return {"ok": False, "reason": "WARMUP", "folds": []}

    chunks = np.array_split(np.array(dates), 4)
    folds = []
    for i in range(1, len(chunks)):
        train_dates = np.concatenate(chunks[:i]) if i > 0 else np.array([])
        test_dates = chunks[i]
        train = df[df[date].isin(train_dates)]
        test = df[df[date].isin(test_dates)]
        if len(train) < MIN_TRAIN_SAMPLES or len(test) < MIN_TEST_SAMPLES:
            continue
        picked, train_m = _select_training_threshold(train, current_threshold)
        active_test = _metrics(test, current_threshold)
        candidate_test = _metrics(test, picked)
        folds.append({
            "train_start": str(pd.Timestamp(train[date].min()).date()),
            "train_end": str(pd.Timestamp(train[date].max()).date()),
            "test_start": str(pd.Timestamp(test[date].min()).date()),
            "test_end": str(pd.Timestamp(test[date].max()).date()),
            "selected_threshold": picked,
            "train": train_m,
            "active_test": active_test,
            "candidate_test": candidate_test,
        })

    if not folds:
        return {"ok": False, "reason": "NO_VALID_FOLDS", "folds": []}

    selected = [f["selected_threshold"] for f in folds]
    stable_threshold = round(float(np.median(selected)), 1)
    oos_dates = [d for f, ch in zip(folds, chunks[1:]) for d in ch]
    oos = df[df[date].isin(oos_dates)].copy()
    active = _metrics(oos, current_threshold)
    candidate = _metrics(oos, stable_threshold)

    return {
        "ok": True,
        "folds": folds,
        "selected_thresholds": selected,
        "stable_threshold": stable_threshold,
        "active_oos": active,
        "candidate_oos": candidate,
    }


def _promotion_allowed(active: Dict, candidate: Dict) -> Tuple[bool, str]:
    if candidate["n"] < MIN_TOTAL_SAMPLES // 2 or active["n"] < MIN_TOTAL_SAMPLES // 2:
        return False, "OOS_SAMPLE_TOO_SMALL"
    wr_lift = (candidate["win_rate"] - active["win_rate"]) / 100.0
    lcb_lift = (candidate["wilson_lcb"] - active["wilson_lcb"]) / 100.0
    if wr_lift < MIN_TEST_WIN_LIFT:
        return False, f"WIN_RATE_LIFT_TOO_SMALL ({wr_lift*100:+.2f}pp)"
    if lcb_lift < MIN_LCB_LIFT:
        return False, f"LCB_LIFT_TOO_SMALL ({lcb_lift*100:+.2f}pp)"
    if candidate.get("pf") is not None and active.get("pf") is not None:
        if candidate["pf"] < max(0.95, active["pf"] * MIN_PF_RATIO):
            return False, f"PF_GUARD ({candidate['pf']:.2f} vs {active['pf']:.2f})"
    if candidate.get("avg_return") is not None and active.get("avg_return") is not None:
        if candidate["avg_return"] < active["avg_return"] - MAX_AVG_RETURN_DROP:
            return False, f"AVG_RETURN_GUARD ({candidate['avg_return']:.3f} vs {active['avg_return']:.3f})"
    return True, f"PROMOTE (+WR {wr_lift*100:.2f}pp | +LCB {lcb_lift*100:.2f}pp)"


def optimize_win_rate(state: Dict, data: pd.DataFrame, *, current_threshold: Optional[float] = None) -> Dict:
    """Run a bounded walk-forward optimization and persist only validated improvements."""
    state = state if isinstance(state, dict) else {}
    current = float(current_threshold if current_threshold is not None else state.get("win_rate_optimizer", {}).get("active_threshold", DEFAULT_THRESHOLD))
    df = _prepare(data)
    result = _walk_forward(df, current)

    wo = state.setdefault("win_rate_optimizer", {})
    wo.setdefault("version", "1.0.0")
    wo["objective"] = "MAX_OOS_WIN_RATE_WITH_RISK_CONSTRAINTS"
    wo["score_col"] = initial_score
    wo["return_col"] = return_d5 if return_d5 in (data.columns if isinstance(data, pd.DataFrame) else []) else None
    wo["outcome_col"] = entry_status if entry_status in (data.columns if isinstance(data, pd.DataFrame) else []) else None
    wo["baseline_threshold"] = round(current, 1)
    wo["last_run"] = datetime.utcnow().isoformat() + "Z"

    if not result.get("ok"):
        wo["status"] = result.get("reason", "WARMUP")
        wo["active_threshold"] = round(current, 1)
        wo["active_oos"] = _metrics(df, current)
        return state

    active = result["active_oos"]
    candidate = result["candidate_oos"]
    proposed = float(result["stable_threshold"])
    allowed, reason = _promotion_allowed(active, candidate)

    wo["walk_forward_folds"] = result["folds"]
    wo["selected_thresholds"] = result["selected_thresholds"]
    wo["candidate_threshold"] = proposed
    wo["active_oos"] = active
    wo["candidate_oos"] = candidate
    wo["decision_reason"] = reason

    if allowed and proposed != round(current, 1):
        wo["active_threshold"] = proposed
        wo["status"] = "PROMOTED"
        wo["promotion_count"] = int(wo.get("promotion_count", 0)) + 1
        wo["last_promotion"] = datetime.utcnow().isoformat() + "Z"
    else:
        wo["active_threshold"] = round(current, 1)
        wo["status"] = "UNCHANGED"

    return state


def summary(state: Dict) -> str:
    wo = state.get("win_rate_optimizer", {}) if isinstance(state, dict) else {}
    a = wo.get("active_oos", {})
    return (
        f"{wo.get('status','WARMUP')} | threshold={wo.get('active_threshold', DEFAULT_THRESHOLD):.1f} "
        f"| OOS WR={a.get('win_rate', 0.0):.2f}% | LCB={a.get('wilson_lcb', 0.0):.2f}% | N={a.get('n',0)}"
    )
