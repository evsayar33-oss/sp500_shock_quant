from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = Path.cwd() if (Path.cwd() / "main.py").exists() else HERE

src_guard = HERE / "autonomy_guard.py"
dst_guard = REPO / "autonomy_guard.py"
if src_guard.resolve() != dst_guard.resolve():
    if dst_guard.exists():
        backup_guard = dst_guard.with_suffix(dst_guard.suffix + ".pre_kurun_backup")
        if not backup_guard.exists():
            shutil.copy2(dst_guard, backup_guard)
    shutil.copy2(src_guard, dst_guard)


def patch(path: Path, old: str, new: str, label: str):
    if not path.exists():
        raise SystemExit(f"MISSING: {path}")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"ANCHOR ERROR {label}: expected 1 match, found {count}")
    backup = path.with_suffix(path.suffix + ".pre_kurun_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new), encoding="utf-8")


main = REPO / "main.py"
patch(
    main,
    "from sp_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA\n",
    "from sp_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA\nfrom autonomy_guard import evaluate_autonomy_guard\n",
    "main import",
)
patch(
    main,
    "def record_clean_ledger_entries(df_scored):\n    bugun_str = datetime.now().strftime('%Y-%m-%d')\n    new_rows = []\n    candidates = df_scored[df_scored['shock_score'] >= 65.0]\n",
    "def record_clean_ledger_entries(df_scored, min_score=65.0):\n    bugun_str = datetime.now().strftime('%Y-%m-%d')\n    new_rows = []\n    candidates = df_scored[df_scored['shock_score'] >= float(min_score)]\n",
    "ledger threshold",
)
patch(
    main,
    "    min_score = 75.0\n    dynamic_weights = {\"vol\": 0.25, \"range\": 0.25, \"flow\": 0.35, \"lambda\": 0.15}\n    if os.path.exists(AI_STATE_FILE):\n",
    '''    min_score = 75.0\n    dynamic_weights = {"vol": 0.25, "range": 0.25, "flow": 0.35, "lambda": 0.15}\n    guard_state = {}\n    if os.path.exists(AI_STATE_FILE):\n''',
    "state initialization",
)
patch(
    main,
    "                dynamic_weights = saved.get('weights', dynamic_weights)\n        except Exception:\n            pass\n\n    df_gecmis = gecmis_veriyi_yukle()\n",
    '''                dynamic_weights = saved.get('weights', dynamic_weights)\n                guard_state = saved if isinstance(saved, dict) else {}\n        except Exception:\n            guard_state = {}\n\n    df_gecmis = gecmis_veriyi_yukle()\n''',
    "state capture",
)
patch(
    main,
    "    df_scored = calculate_shock_scores(df_current, df_gecmis, dynamic_weights=dynamic_weights)\n    \n    if df_scored.empty:\n        return\n",
    '''    df_scored = calculate_shock_scores(df_current, df_gecmis, dynamic_weights=dynamic_weights)\n\n    if df_scored.empty:\n        return\n\n    ledger_returns = pd.DataFrame()\n    if os.path.exists(LEDGER_FILE):\n        try:\n            ledger_returns = pd.read_csv(LEDGER_FILE)\n        except Exception:\n            ledger_returns = pd.DataFrame()\n    guard_result = evaluate_autonomy_guard(\n        guard_state,\n        features=df_scored,\n        performance_returns=(ledger_returns["return_d5"] if "return_d5" in ledger_returns.columns else None),\n        data_quality_score=100.0,\n        row_count=len(df_current),\n        min_rows=100,\n        project="sp500_shock",\n    )\n    effective_min_score = float(min_score) + float(guard_result.get("signal_threshold_add", 0.0))\n    if guard_result.get("block_new_entries"):\n        effective_min_score = 101.0\n    guard_state["autonomy_guard"] = guard_state.get("autonomy_guard", {})\n    with open(AI_STATE_FILE, "w", encoding="utf-8") as f:\n        json.dump(guard_state, f, indent=4, ensure_ascii=False)\n''',
    "main guard",
)
patch(
    main,
    "    record_clean_ledger_entries(df_scored)\n\n    # 4. Raporu ilet\n    telegram_msg = format_shock_report(df_scored, exit_signals_text, min_score)\n",
    "    record_clean_ledger_entries(df_scored, effective_min_score)\n\n    # 4. Raporu ilet\n    telegram_msg = format_shock_report(df_scored, exit_signals_text, effective_min_score)\n",
    "main apply threshold",
)

# Evening auditor must merge into the existing AI state instead of replacing it with a
# small legacy dict; otherwise the autonomy controller would be erased every evening.
aud = REPO / "sp_auditor.py"
patch(
    aud,
    "from sp_fetcher import get_sp500_raw_data\n",
    "from sp_fetcher import get_sp500_raw_data\nfrom autonomy_guard import evaluate_autonomy_guard\n",
    "auditor import",
)
patch(
    aud,
    "    df_ledger = update_ledger_returns(df_close)\n    completed = df_ledger[df_ledger['is_completed'] == 1] if not df_ledger.empty else pd.DataFrame()\n    completed_count = len(completed)\n\n    backtest_msg = \"\"\n",
    '''    df_ledger = update_ledger_returns(df_close)\n    completed = df_ledger[df_ledger['is_completed'] == 1] if not df_ledger.empty else pd.DataFrame()\n    completed_count = len(completed)\n\n    state = {}\n    if os.path.exists(AI_STATE_FILE):\n        try:\n            with open(AI_STATE_FILE, 'r', encoding='utf-8') as f:\n                loaded = json.load(f)\n            state = loaded if isinstance(loaded, dict) else {}\n        except Exception:\n            state = {}\n\n    backtest_msg = ""\n''',
    "auditor state load",
)
patch(
    aud,
    '''            new_state = {\n                "thresholds": {"th_vol": 1.5, "th_range": 1.4, "th_flow": 2.0, "th_lambda": 1.0, "min_score": best_p['min_score']},\n                "weights": {"vol": best_p['vol'], "range": best_p['range'], "flow": best_p['flow'], "lambda": best_p['lambda']},\n                "status": f"🏆 REJİM KORUMALI BACKTEST ONAYLI (Win Rate: %{win_r:.1f})"\n            }\n            with open(AI_STATE_FILE, 'w') as f:\n                json.dump(new_state, f, indent=4)\n''',
    '''            # Merge legacy calibration into the existing state; never erase autonomy_guard.\n            state["thresholds"] = {"th_vol": 1.5, "th_range": 1.4, "th_flow": 2.0, "th_lambda": 1.0, "min_score": best_p['min_score']}\n            state["weights"] = {"vol": best_p['vol'], "range": best_p['range'], "flow": best_p['flow'], "lambda": best_p['lambda']}\n            state["status"] = f"🏆 REJİM KORUMALI BACKTEST ONAYLI (Win Rate: %{win_r:.1f})"\n''',
    "auditor state merge",
)
patch(
    aud,
    "    rep = \"🔬 <b>S&P 500 GÜVENLİK KALKANLI DENETİM RAPORU</b>\\n\"\n",
    '''    guard_result = evaluate_autonomy_guard(\n        state,\n        features=None,\n        performance_returns=(completed["return_d5"] if "return_d5" in completed.columns else None),\n        data_quality_score=100.0,\n        row_count=len(df_close),\n        min_rows=100,\n        project="sp500_shock",\n    )\n    with open(AI_STATE_FILE, 'w', encoding='utf-8') as f:\n        json.dump(state, f, indent=4, ensure_ascii=False)\n\n    rep = "🔬 <b>S&P 500 GÜVENLİK KALKANLI DENETİM RAPORU</b>\\n"\n    rep += f"🛡️ <b>Otonomi:</b> {guard_result.get('mode', 'NORMAL')} | x{guard_result.get('exposure_multiplier', 1.0):.2f} | {guard_result.get('reason', '')}\\n"\n''',
    "auditor guard save/report",
)

for name in ("autonomy_guard.py", "main.py", "sp_auditor.py"):
    subprocess.check_call([sys.executable, "-m", "py_compile", str(REPO / name)])
print("OK: S&P 500 Kur-Unut guard installed. Backups: *.pre_kurun_backup")
