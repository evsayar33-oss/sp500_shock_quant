import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime

from sp_fetcher import fetch_all_data
from sp_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA

AI_STATE_FILE = "sp500_ai_state.json"
LEDGER_FILE = "backtest_ledger.csv"

def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('CHAT_ID')
    if not token or not chat_id:
        print("Telegram kimlik bilgileri eksik.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 3800
    messages = []
    if len(message) > max_len:
        parts = message.split("\n\n")
        current_msg = ""
        for p in parts:
            if len(current_msg) + len(p) + 2 < max_len:
                current_msg += p + "\n\n"
            else:
                messages.append(current_msg.strip())
                current_msg = p + "\n\n"
        if current_msg:
            messages.append(current_msg.strip())
    else:
        messages = [message]

    for idx, msg in enumerate(messages):
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            res = requests.post(url, json=payload, timeout=15)
            if not res.json().get("ok"):
                payload.pop("parse_mode")
                requests.post(url, json=payload, timeout=15)
        except Exception as e:
            print(f"Telegram hatası: {e}")

def record_clean_ledger_entries(df_scored):
    bugun_str = datetime.now().strftime('%Y-%m-%d')
    new_rows = []
    candidates = df_scored[df_scored['shock_score'] >= 65.0]
    for _, r in candidates.iterrows():
        new_rows.append({
            "date": bugun_str,
            "ticker": r['ticker'],
            "entry_price": r['close'],
            "z_vol": r.get('z_vol', 0.0),
            "z_range": r.get('z_range', 0.0),
            "z_flow": r.get('z_flow', 0.0),
            "z_lambda": r.get('z_lambda', 0.0),
            "is_above_trend": 1 if r.get('is_above_trend') else 0,
            "entry_status": r.get('entry_status', 'NORMAL'),
            "initial_score": r.get('shock_score', 0.0),
            "price_d1": np.nan,
            "price_d3": np.nan,
            "price_d5": np.nan,
            "return_d1": np.nan,
            "return_d3": np.nan,
            "return_d5": np.nan,
            "is_completed": 0
        })

    if not new_rows:
        return

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(LEDGER_FILE):
        try:
            df_old = pd.read_csv(LEDGER_FILE)
            df_old = df_old[~((df_old['date'] == bugun_str) & (df_old['ticker'].isin(df_new['ticker'])))]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_csv(LEDGER_FILE, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] S&P 500 Defterine {len(df_new)} temiz işlem kaydedildi.")

def format_shock_report(df_scored, min_score=75.0):
    shocks = df_scored[df_scored['shock_score'] >= min_score].sort_values(by='shock_score', ascending=False)
    
    msg = f"🗽 <b>S&P 500 QUANT MOMENTUM LİSTESİ ({min_score:.1f}+)</b>\n"
    msg += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | New York Seans Açılışı</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if shocks.empty:
        msg += f"ℹ️ <i>Bugün {min_score:.1f} puan ve üzeri kurumsal akış alan S&P 500 hissesi bulunamadı.</i>"
        return msg

    for idx, row in shocks.head(15).iterrows():
        trend_icon = "EMA20/SMA50 Üstü ✅" if row.get('is_above_trend') else "Ortalama Altı ⚠️"
        msg += f"🚀 <b>#{row['ticker']}</b> ── <b>{row['shock_score']:.1f} Puan</b> ({row['stars']})\n"
        msg += f"• <b>Fiyat:</b> ${row['close']:.2f} | <b>Değişim:</b> %{row['change_%']:+.2f}\n"
        msg += f"• <b>Bölge:</b> <i>{row['entry_status']}</i>\n"
        msg += f"• <b>Trend:</b> <i>{trend_icon}</i>\n"
        msg += f"💰 <b>KASA ÖNERİSİ:</b> <b>{row['allocation']}</b>\n\n"
        
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🎯 <i>Toplam {len(shocks)} adet yüksek güvenli hisse tespit edildi.</i>\n\n"
    msg += "🛑 <b>RİSK KURALI:</b> <i>Stop-Loss seviyesini 09:30 - 09:45 açılış barının en dibine koyunuz!</i>"
    return msg

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === S&P 500 Coordinated Shock Scanner Başlıyor ===")
    
    df_current = fetch_all_data()
    if df_current.empty:
        print("Hata: S&P 500 verisi alınamadı.")
        return

    # Kayıtlı parametreleri oku
    min_score = 75.0
    dynamic_weights = {"vol": 0.25, "range": 0.25, "flow": 0.35, "lambda": 0.15}
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, 'r') as f:
                saved = json.load(f)
                min_score = saved.get('thresholds', {}).get('min_score', 75.0)
                dynamic_weights = saved.get('weights', dynamic_weights)
        except Exception:
            pass

    df_gecmis = gecmis_veriyi_yukle()
    df_scored = calculate_shock_scores(df_current, df_gecmis, dynamic_weights=dynamic_weights)
    
    if df_scored.empty:
        return

    record_clean_ledger_entries(df_scored)

    if not df_gecmis.empty:
        bugun = pd.Timestamp.now().normalize()
        df_gecmis = df_gecmis[df_gecmis['tarih'] != bugun]
        df_yeni = pd.concat([df_gecmis, df_scored], ignore_index=True)
    else:
        df_yeni = df_scored

    df_yeni['tarih'] = pd.to_datetime(df_yeni['tarih'])
    limit_tarih = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    df_yeni = df_yeni[df_yeni['tarih'] >= limit_tarih]
    df_yeni.to_csv(GECMIS_DOSYA, index=False)

    telegram_msg = format_shock_report(df_scored, min_score)
    send_telegram_message(telegram_msg)
    print("S&P 500 Açılış taraması Telegram'a iletildi.")

if __name__ == "__main__":
    main()
