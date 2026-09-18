"""Regime Stress-Test / Drift / Safe-Mode guardrail.

Pure numpy/pandas implementation. It does not replace the strategy/scoring engine.
It observes the current engine, detects distribution/performance/ops stress, and
controls exposure / entry gating through an explicit state machine.

Production principle:
- No synthetic market facts are used in live scoring.
- Synthetic data in run_stress_test() is internal unit-test data only.
- Guard failures are fail-closed for new entries, never a Python crash.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

AUTONOMY_VERSION = "1.0.0"

DEFAULT_GUARD_STATE = {
    "version": AUTONOMY_VERSION,
    "mode": "NORMAL",  # NORMAL / WATCH / SAFE / RECOVERY
    "exposure_multiplier": 1.0,
    "signal_threshold_add": 0.0,
    "block_new_entries": False,
    "drift_score": 0.0,
    "performance_drift": 0.0,
    "ops_score": 100.0,
    "regime_stress": 0.0,
    "risk_score": 0.0,
    "watch_streak": 0,
    "safe_streak": 0,
    "recovery_streak": 0,
    "healthy_streak": 0,
    "last_transition": None,
    "last_evaluation": None,
    "reason": "BOOTSTRAP",
    "recent_regimes": [],
    "feature_baseline": {},
    "stress_test": {"passed": False, "timestamp": None, "version": AUTONOMY_VERSION},
}

MODE_POLICY = {
    "NORMAL": {"exposure_multiplier": 1.00, "signal_threshold_add": 0.0, "block_new_entries": False},
    "WATCH": {"exposure_multiplier": 0.60, "signal_threshold_add": 5.0, "block_new_entries": False},
    "SAFE": {"exposure_multiplier": 0.00, "signal_threshold_add": 100.0, "block_new_entries": True},
    "RECOVERY": {"exposure_multiplier": 0.35, "signal_threshold_add": 8.0, "block_new_entries": False},
}

DEFAULT_FEATURES = {
    "orderflow": [
        "pre_move_score", "flow_score", "resilience_score", "risk_score",
        "quality_score", "overnight_risk", "rvol", "perf_w", "perf_1m",
    ],
    "bist_shock": [
        "shock_score", "non_price_score", "flow_score", "activity_score",
        "liquidity_score", "resilience_score", "overnight_risk", "z_vol", "z_flow", "z_range",
    ],
    "sp500_shock": [
        "shock_score", "z_vol", "z_flow", "z_range", "z_lambda",
        "change_%", "rvol", "perf_1m", "perf_3m",
    ],
}


def _num(value, default=0.0):
    try:
        x = float(value)
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def _clip01(value):
    return float(np.clip(_num(value, 0.0), 0.0, 1.0))


def _merge(base, incoming):
    out = deepcopy(base)
    if isinstance(incoming, Mapping):
        for k, v in incoming.items():
            if isinstance(v, Mapping) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = deepcopy(v)
    return out


def _safe_series(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def _robust_stats(s: pd.Series):
    s = _safe_series(s)
    if s.empty:
        return None
    q25, med, q75 = np.percentile(s.to_numpy(), [25, 50, 75])
    mad = float(np.median(np.abs(s.to_numpy() - med)))
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 6),
        "std": round(float(s.std(ddof=0)), 6),
        "median": round(float(med), 6),
        "q25": round(float(q25), 6),
        "q75": round(float(q75), 6),
        "mad": round(mad, 6),
    }


def _baseline_scale(ref):
    iqr = max(_num(ref.get("q75"), 0.0) - _num(ref.get("q25"), 0.0), 0.0)
    mad = _num(ref.get("mad"), 0.0) * 1.4826
    std = _num(ref.get("std"), 0.0)
    # At least a tiny scale. Using the largest robust scale reduces false drift flags
    # when a feature is naturally noisy.
    return max(mad, iqr / 1.349, std * 0.5, 1e-6)


def _feature_drift(ref, cur):
    if not ref or not cur:
        return 0.0
    scale = _baseline_scale(ref)
    med_shift = abs(_num(cur.get("median")) - _num(ref.get("median"))) / scale
    cur_iqr = max(_num(cur.get("q75")) - _num(cur.get("q25")), 1e-9)
    ref_iqr = max(_num(ref.get("q75")) - _num(ref.get("q25")), 1e-9)
    dispersion_shift = abs(math.log(cur_iqr / ref_iqr))
    missing = max(0.0, 1.0 - min(_num(cur.get("n"), 0.0) / max(_num(ref.get("n"), 1.0), 1.0), 1.0))
    shift_component = min(med_shift / 4.0, 1.0)
    dispersion_component = min(dispersion_shift / 1.5, 1.0)
    return float(np.clip(0.65 * shift_component + 0.30 * dispersion_component + 0.05 * missing, 0.0, 1.0))


def _update_reference(ref, cur, alpha=0.05):
    if not ref:
        return deepcopy(cur)
    out = deepcopy(ref)
    for key in ("mean", "std", "median", "q25", "q75", "mad"):
        out[key] = round((1.0 - alpha) * _num(ref.get(key)) + alpha * _num(cur.get(key)), 6)
    out["n"] = int(min(_num(ref.get("n"), 0) + _num(cur.get("n"), 0), 5000))
    return out


def _extract_regime(regime):
    if regime is None:
        return "UNKNOWN"
    if isinstance(regime, Mapping):
        return str(regime.get("label", regime.get("regime", "UNKNOWN"))).upper()
    return str(regime).upper()


def infer_generic_regime(df: Optional[pd.DataFrame]) -> tuple[str, float]:
    if df is None or df.empty or "change_%" not in df.columns:
        return "UNKNOWN", 0.0
    daily = _safe_series(df["change_%"])
    if daily.empty:
        return "UNKNOWN", 0.0
    up = float((daily > 0).mean())
    down = float((daily < 0).mean())
    med = float(daily.median())
    mean = float(daily.mean())
    disp = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0
    if down >= 0.70 or med <= -1.2 or mean <= -1.8:
        return "CRASH", round(min(0.98, 0.65 + max(down - 0.70, 0.0)), 3)
    if down >= 0.60 or med <= -0.45:
        return "STRESS", round(min(0.95, 0.55 + max(down - 0.60, 0.0)), 3)
    if up >= 0.62 and med >= 0.30 and mean >= 0.55:
        return "EXPANSION", round(min(0.95, 0.55 + max(up - 0.62, 0.0)), 3)
    if disp >= max(2.25, abs(mean) * 1.8 + 1.0) and 0.35 <= down <= 0.65:
        return "ROTATION", 0.70
    if disp <= 1.20 and abs(med) <= 0.35:
        return "QUIET", 0.75
    return "NORMAL", 0.50


def _regime_stress(recent_regimes, current, confidence):
    history = [str(x).upper() for x in recent_regimes[-6:] if x]
    history.append(str(current).upper())
    stress = 0.0
    if history and history[-1] != history[-2] if len(history) >= 2 else False:
        stress += 0.25
    uniq5 = len(set(history[-5:]))
    uniq3 = len(set(history[-3:]))
    if uniq5 >= 3:
        stress += 0.30
    if uniq3 >= 3:
        stress += 0.25
    if _clip01(confidence) < 0.45:
        stress += 0.20
    return float(np.clip(stress, 0.0, 1.0))


def _performance_drift(returns, recent_n=20, baseline_n=60):
    s = _safe_series(returns)
    if len(s) < recent_n + 10:
        return 0.0, {"samples": int(len(s)), "status": "WARMUP"}
    recent = s.iloc[-recent_n:]
    base = s.iloc[-min(baseline_n, len(s) - recent_n):-recent_n]
    if len(base) < 10:
        return 0.0, {"samples": int(len(s)), "status": "WARMUP"}

    recent_wr = float((recent > 0).mean())
    base_wr = float((base > 0).mean())
    recent_mean = float(recent.mean())
    base_mean = float(base.mean())
    scale = max(float(base.abs().median()), 0.25)
    mean_decay = max(0.0, (base_mean - recent_mean) / (3.0 * scale))
    wr_decay = max(0.0, (base_wr - recent_wr) / 0.25)
    drift = float(np.clip(0.60 * mean_decay + 0.40 * wr_decay, 0.0, 1.0))
    return drift, {
        "samples": int(len(s)),
        "recent_n": int(len(recent)),
        "baseline_n": int(len(base)),
        "recent_mean": round(recent_mean, 4),
        "baseline_mean": round(base_mean, 4),
        "recent_win_rate": round(recent_wr * 100.0, 2),
        "baseline_win_rate": round(base_wr * 100.0, 2),
    }


def _ops_score(data_quality_score, row_count=None, min_rows=30):
    dq = float(np.clip(_num(data_quality_score, 0.0), 0.0, 100.0))
    if row_count is None:
        row_health = 100.0
    else:
        row_health = 100.0 * float(np.clip(_num(row_count, 0.0) / max(min_rows, 1), 0.0, 1.0))
    return round(0.75 * dq + 0.25 * row_health, 2)


def _stress_case(expected_mode, drift, perf, ops, regime_stress, prev_mode="NORMAL", healthy=False):
    # Internal deterministic state transition model used only by the unit test.
    bad = (ops < 50.0) or drift >= 0.75 or perf >= 0.80 or regime_stress >= 0.90
    watch = (ops < 75.0) or drift >= 0.45 or perf >= 0.45 or regime_stress >= 0.60
    if prev_mode == "SAFE":
        mode = "RECOVERY" if healthy and ops >= 85 and drift < 0.30 and perf < 0.35 else "SAFE"
    elif prev_mode == "RECOVERY":
        mode = "SAFE" if bad else ("NORMAL" if healthy and not watch else "RECOVERY")
    elif bad:
        mode = "SAFE"
    else:
        mode = "WATCH" if watch else "NORMAL"
    return mode == expected_mode


def run_stress_test() -> dict:
    """Deterministic unit test of the guard state machine. No market facts are emitted."""
    cases = [
        ("NORMAL", 0.10, 0.10, 98.0, 0.05, "NORMAL", True),
        ("WATCH", 0.52, 0.10, 96.0, 0.10, "NORMAL", False),
        ("SAFE", 0.85, 0.10, 95.0, 0.10, "WATCH", False),
        ("SAFE", 0.20, 0.85, 95.0, 0.15, "WATCH", False),
        ("SAFE", 0.20, 0.20, 45.0, 0.95, "WATCH", False),
        ("RECOVERY", 0.20, 0.20, 90.0, 0.20, "SAFE", True),
        ("NORMAL", 0.05, 0.05, 98.0, 0.05, "RECOVERY", True),
    ]
    results = [_stress_case(*case) for case in cases]
    passed = bool(all(results))
    return {
        "passed": passed,
        "version": AUTONOMY_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": int(len(cases)),
        "passed_cases": int(sum(results)),
    }


def evaluate_autonomy_guard(
    state: dict,
    *,
    features: Optional[pd.DataFrame] = None,
    regime=None,
    regime_confidence=0.0,
    performance_returns=None,
    data_quality_score=100.0,
    row_count=None,
    min_rows=30,
    project="generic",
    feature_columns: Optional[Iterable[str]] = None,
    stress_test_due=True,
) -> dict:
    """Evaluate and persist the autonomous safety controller into state['autonomy_guard']."""
    try:
        if not isinstance(state, dict):
            state = {}
        guard = _merge(DEFAULT_GUARD_STATE, state.get("autonomy_guard", {}))

        if stress_test_due or not guard.get("stress_test", {}).get("passed"):
            guard["stress_test"] = run_stress_test()

        # Infer regime for engines that do not expose a native classifier.
        if regime is None:
            regime_label, inferred_conf = infer_generic_regime(features)
            confidence = inferred_conf
        else:
            regime_label = _extract_regime(regime)
            confidence = _clip01(regime_confidence)

        prior_regimes = [str(x).upper() for x in guard.get("recent_regimes", []) if x]
        regime_stress = _regime_stress(prior_regimes, regime_label, confidence)

        cols = list(feature_columns or DEFAULT_FEATURES.get(project, []))
        baseline = guard.get("feature_baseline", {})
        drifts = []
        current_stats = {}
        if features is not None and not features.empty:
            for col in cols:
                if col not in features.columns:
                    continue
                cur = _robust_stats(features[col])
                if not cur:
                    continue
                current_stats[col] = cur
                ref = baseline.get(col)
                if ref:
                    drifts.append(_feature_drift(ref, cur))
                # Baseline learns slowly. Do not absorb a suspected drift event into
                # the reference distribution while drift is elevated.
                if guard.get("mode") != "SAFE" and (not ref or _feature_drift(ref, cur) < 0.35):
                    baseline[col] = _update_reference(ref, cur, alpha=0.05)
        if drifts:
            # A large shift in even one critical feature must not disappear inside a
            # cross-sectional average. Use max-dominant aggregation.
            drift_score = float(np.clip(0.70 * max(drifts) + 0.30 * np.mean(drifts), 0.0, 1.0))
        else:
            drift_score = 0.0

        perf_drift, perf_detail = _performance_drift(performance_returns)
        ops = _ops_score(data_quality_score, row_count=row_count, min_rows=min_rows)
        ops_risk = 1.0 - ops / 100.0
        risk_score = float(np.clip(max(drift_score, perf_drift, ops_risk, regime_stress), 0.0, 1.0))

        stress_test_failed = not bool(guard.get("stress_test", {}).get("passed"))
        severe = (
            stress_test_failed or
            ops < 50.0 or
            drift_score >= 0.75 or
            perf_drift >= 0.80 or
            (regime_stress >= 0.90 and confidence < 0.45)
        )
        watch = (
            stress_test_failed or
            ops < 75.0 or
            drift_score >= 0.45 or
            perf_drift >= 0.45 or
            regime_stress >= 0.60
        )
        healthy = (
            ops >= 90.0 and
            drift_score < 0.30 and
            perf_drift < 0.35 and
            regime_stress < 0.50 and
            bool(guard.get("stress_test", {}).get("passed"))
        )

        mode = str(guard.get("mode", "NORMAL")).upper()
        old_mode = mode
        guard["watch_streak"] = int(guard.get("watch_streak", 0)) + (1 if watch else 0)
        guard["safe_streak"] = int(guard.get("safe_streak", 0)) + (1 if severe else 0)
        guard["healthy_streak"] = int(guard.get("healthy_streak", 0)) + (1 if healthy else 0)

        if mode == "NORMAL":
            if severe:
                mode = "SAFE" if guard["safe_streak"] >= 2 or ops < 50.0 else "WATCH"
            elif watch and guard["watch_streak"] >= 2:
                mode = "WATCH"
        elif mode == "WATCH":
            if severe:
                mode = "SAFE" if guard["safe_streak"] >= 2 or ops < 50.0 else "WATCH"
            elif not watch and healthy:
                mode = "NORMAL"
        elif mode == "SAFE":
            if healthy:
                guard["recovery_streak"] = int(guard.get("recovery_streak", 0)) + 1
            else:
                guard["recovery_streak"] = 0
            if guard["recovery_streak"] >= 3:
                mode = "RECOVERY"
        elif mode == "RECOVERY":
            if severe:
                mode = "SAFE"
                guard["recovery_streak"] = 0
            elif healthy:
                guard["recovery_streak"] = int(guard.get("recovery_streak", 0)) + 1
                if guard["recovery_streak"] >= 6:
                    mode = "NORMAL"
            else:
                guard["recovery_streak"] = 0

        # Any internal guard-test failure or hard operational failure blocks new entries.
        if stress_test_failed or ops < 40.0:
            mode = "SAFE"

        policy = MODE_POLICY[mode]
        guard["mode"] = mode
        guard["exposure_multiplier"] = policy["exposure_multiplier"]
        guard["signal_threshold_add"] = policy["signal_threshold_add"]
        guard["block_new_entries"] = policy["block_new_entries"]
        guard["drift_score"] = round(drift_score, 4)
        guard["performance_drift"] = round(perf_drift, 4)
        guard["ops_score"] = round(ops, 2)
        guard["regime_stress"] = round(regime_stress, 4)
        guard["risk_score"] = round(risk_score, 4)
        guard["recent_regimes"] = (prior_regimes + [regime_label])[-10:]
        guard["feature_baseline"] = baseline
        guard["last_evaluation"] = datetime.now(timezone.utc).isoformat()
        if mode != old_mode:
            guard["last_transition"] = guard["last_evaluation"]
        reasons = []
        if severe: reasons.append("SEVERE_STRESS")
        elif watch: reasons.append("WATCH_STRESS")
        if drift_score >= 0.45: reasons.append("FEATURE_DRIFT")
        if perf_drift >= 0.45: reasons.append("PERFORMANCE_DRIFT")
        if ops < 75.0: reasons.append("OPS_DATA_QUALITY")
        if regime_stress >= 0.60: reasons.append("REGIME_TRANSITION_STRESS")
        if not reasons: reasons.append("HEALTHY")
        guard["reason"] = "+".join(reasons)
        guard["performance_detail"] = perf_detail

        state["autonomy_guard"] = guard
        state.setdefault("autonomy", {})
        state["autonomy"]["version"] = AUTONOMY_VERSION
        state["autonomy"]["last_mode"] = mode
        state["autonomy"]["last_reason"] = guard["reason"]

        return {
            "mode": mode,
            "exposure_multiplier": policy["exposure_multiplier"],
            "signal_threshold_add": policy["signal_threshold_add"],
            "block_new_entries": policy["block_new_entries"],
            "reason": guard["reason"],
            "drift_score": drift_score,
            "performance_drift": perf_drift,
            "ops_score": ops,
            "regime_stress": regime_stress,
            "risk_score": risk_score,
            "regime": regime_label,
            "regime_confidence": confidence,
            "performance_detail": perf_detail,
            "stress_test_passed": bool(guard["stress_test"].get("passed")),
        }
    except Exception as exc:
        # Fail closed rather than crashing production scan.
        guard = _merge(DEFAULT_GUARD_STATE, state.get("autonomy_guard", {}) if isinstance(state, dict) else {})
        guard.update({
            "mode": "SAFE",
            "exposure_multiplier": 0.0,
            "signal_threshold_add": 100.0,
            "block_new_entries": True,
            "reason": "GUARD_ERROR_SAFE_MODE",
            "last_evaluation": datetime.now(timezone.utc).isoformat(),
        })
        if isinstance(state, dict):
            state["autonomy_guard"] = guard
        return {
            "mode": "SAFE",
            "exposure_multiplier": 0.0,
            "signal_threshold_add": 100.0,
            "block_new_entries": True,
            "reason": f"GUARD_ERROR_SAFE_MODE:{type(exc).__name__}",
            "drift_score": 1.0,
            "performance_drift": 1.0,
            "ops_score": 0.0,
            "regime_stress": 1.0,
            "risk_score": 1.0,
            "regime": "UNKNOWN",
            "regime_confidence": 0.0,
            "performance_detail": {"status": "ERROR"},
            "stress_test_passed": False,
        }


def apply_entry_guard(df: pd.DataFrame, score_col: str, base_threshold: float, guard_result: Mapping, eligible_col="eligible") -> pd.DataFrame:
    """Apply guard policy without mutating underlying scores."""
    if df is None or df.empty:
        return df
    out = df.copy()
    threshold = float(base_threshold) + _num(guard_result.get("signal_threshold_add"), 0.0)
    if guard_result.get("block_new_entries"):
        if eligible_col in out.columns:
            out[eligible_col] = False
        else:
            out["eligible"] = False
        return out
    mask = pd.to_numeric(out.get(score_col), errors="coerce") >= threshold
    if eligible_col in out.columns:
        out[eligible_col] = out[eligible_col].astype(bool) & mask
    else:
        out["eligible"] = mask
    return out


def scale_allocation_text(text: str, multiplier: float) -> str:
    """Conservative display-only scaling for human-readable allocation strings."""
    try:
        import re
        m = float(np.clip(_num(multiplier, 1.0), 0.0, 1.0))
        def repl(match):
            val = float(match.group(1))
            return f"%{val * m:.1f}"
        return re.sub(r"%(\d+(?:\.\d+)?)", repl, str(text))
    except Exception:
        return str(text)
