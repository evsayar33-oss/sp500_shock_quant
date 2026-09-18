import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime
from sp_fetcher import get_sp500_raw_data
from autonomy_guard import evaluate_autonomy_guard

AI_STATE_FILE = "sp500_ai_state.json"
LEDGER_FILE = "backtest_ledger.csv"
MIN_BACKTEST_SAMPLES = 25

def send_telegram_audit(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('CHAT_ID')
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        res = requests.post(url, json=payload, timeout=15)
        if not res.json().get("ok"):
            payload.pop("parse_mode")
            requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"Hata: {e}")

def update_ledger_returns(df_close):
    if not os.path.exists(LEDGER_FILE):
        return pd.DataFrame()

    df_ledger = pd.read_csv(LEDGER_FILE)
    if df_ledger.empty:
        return df_ledger

    close_map = dict(zip(df_close['ticker'], df_close['close']))
    bugun = datetime.now().date()

    for idx, row in df_ledger.iterrows():
        ticker = row['ticker']
        if ticker not in close_map:
            continue

        curr_p = close_map[ticker]
        entry_p = float(row['entry_price'])
        row_date = datetime.strptime(str(row['date']), '%Y-%m-%d').date()
        days_passed = (bugun - row_date).days

        if days_passed >= 1 and pd.isna(row['price_d1']):
            df_ledger.at[idx, 'price_d1'] = curr_p
            df_ledger.at[idx, 'return_d1'] = round(((curr_p - entry_p) / entry_p) * 100, 2)

        if days_passed >= 3 and pd.isna(row['price_d3']):
            df_ledger.at[idx, 'price_d3'] = curr_p
            df_ledger.at[idx, 'return_d3'] = round(((curr_p - entry_p) / entry_p) * 100, 2)

        if days_passed >= 5 and pd.isna(row['price_d5']):
            df_ledger.at[idx, 'price_d5'] = curr_p
            df_ledger.at[idx, 'return_d5'] = round(((curr_p - entry_p) / entry_p) * 100, 2)
            df_ledger.at[idx, 'is_completed'] = 1

    df_ledger.to_csv(LEDGER_FILE, index=False)
    return df_ledger

def run_grid_search_backtest(completed_trades):
    param_grid = [
        {"vol": 0.25, "range": 0.25, "flow": 0.35, "lambda": 0.15, "min_score": 75.0},
        {"vol": 0.20, "range": 0.25, "flow": 0.40, "lambda": 0.15, "min_score": 76.0},
        {"vol": 0.18, "range": 0.22, "flow": 0.45, "lambda": 0.15, "min_score": 77.0},
        {"vol": 0.28, "range": 0.22, "flow": 0.35, "lambda": 0.15, "min_score": 74.0},
    ]

    best_score = -999.0
    best_params = None
    best_win_rate = 0.0

    for params in param_grid:
        gains = []
        losses = []

        for _, tr in completed_trades.iterrows():
            sim_score = (
                tr['z_vol'] * params['vol'] +
                tr['z_range'] * params['range'] +
                tr['z_flow'] * params['flow'] +
                tr['z_lambda'] * params['lambda']
            ) * 15.0

            if sim_score >= params['min_score']:
                ret = tr['return_d5']
                if ret > 0:
                    gains.append(ret)
                else:
                    losses.append(abs(ret))

        total_trades = len(gains) + len(losses)
        if total_trades >= 5:
            win_rate = (len(gains) / total_trades) * 100.0
            sum_gains = sum(gains)
            sum_losses = sum(losses) if sum(losses) > 0 else 1.0
            profit_factor = sum_gains / sum_losses
            eval_metric = profit_factor * (win_rate / 100.0)
            if eval_metric > best_score:
                best_score = eval_metric
                best_params = params
                best_win_rate = win_rate

    return best_params, best_win_rate

def run_evening_audit():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] S&P 500 Kapanış Denetimi Başlatılıyor...")
    try:
        df_close = get_sp500_raw_data()
    except:
        df_close = pd.DataFrame()

    if df_close.empty:
        print("Kapanış verisi alınamadı.")
        return

    df_ledger = update_ledger_returns(df_close)
    completed = df_ledger[df_ledger['is_completed'] == 1] if not df_ledger.empty else pd.DataFrame()
    completed_count = len(completed)

    state = {}
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            state = loaded if isinstance(loaded, dict) else {}
        except Exception:
            state = {}

    backtest_msg = ""
    if completed_count >= MIN_BACKTEST_SAMPLES:
        best_p, win_r = run_grid_search_backtest(completed)
        if best_p:
            # Merge legacy calibration into the existing state; autonomy_guard is preserved.
            state["thresholds"] = {"th_vol": 1.5, "th_range": 1.4, "th_flow": 2.0, "th_lambda": 1.0, "min_score": best_p['min_score']}
            state["weights"] = {"vol": best_p['vol'], "range": best_p['range'], "flow": best_p['flow'], "lambda": best_p['lambda']}
            state["status"] = f"🏆 REJİM KORUMALI BACKTEST ONAYLI (Win Rate: %{win_r:.1f})"
            backtest_msg = (
                f"🧪 <b>S&P 500 ÇOKLU REJİM BACKTEST SONUÇLANDI:</b>\n"
                f"• <i>{completed_count} işlem üzerinde simüle edildi.</i>\n"
                f"• Hedef Baraj: <b>{best_p['min_score']:.1f}</b> | Akış Ağırlığı: <b>%{best_p['flow']*100:.0f}</b>\n"
                f"• Kazanma Oranı: <b>%{win_r:.1f}</b>\n"
            )
    else:
        kalan = MIN_BACKTEST_SAMPLES - completed_count
        backtest_msg = (
            f"⏳ <b>BACKTEST DEFTER İLERLEMESİ:</b>\n"
            f"• <i>Temizlenen: <b>{completed_count} / {MIN_BACKTEST_SAMPLES} Tamamlanmış İşlem</b></i>\n"
            f"• <i>Kalan {kalan} işlem sonra rejim simülasyonu otomatik çalışacaktır.</i>\n"
        )

    guard_result = evaluate_autonomy_guard(
        state,
        features=None,
        performance_returns=(completed["return_d5"] if "return_d5" in completed.columns else None),
        data_quality_score=100.0,
        row_count=len(df_close),
        min_rows=100,
        project="sp500_shock",
    )
    with open(AI_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4, ensure_ascii=False)

    rep = "🔬 <b>S&P 500 GÜVENLİK KALKANLI DENETİM RAPORU</b>\n"
    rep += f"🛡️ <b>Otonomi:</b> {guard_result.get('mode', 'NORMAL')} | x{guard_result.get('exposure_multiplier', 1.0):.2f} | {guard_result.get('reason', '')}\n"
    rep += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | New York Kapanışı</i>\n"
    rep += "━━━━━━━━━━━━━━━━━━━━\n\n"
    rep += backtest_msg
    send_telegram_audit(rep)
    print("S&P 500 Denetim raporu iletildi.")

if __name__ == "__main__":
    run_evening_audit()
