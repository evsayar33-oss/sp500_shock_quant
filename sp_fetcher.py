import requests
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# TradingView çökerse devreye girecek S&P ve ABD piyasasının en likit çekirdek hisse sepeti
BACKUP_WATCHLIST = [
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "PLTR", "AVGO",
    "NFLX", "SMCI", "ARM", "COIN", "INTC", "MU", "QCOM", "TXN", "AMAT", "LRCX",
    "JPM", "V", "MA", "UNH", "LLY", "NVO", "XOM", "CVX", "GE", "CAT",
    "BA", "UBER", "ABNB", "DASH", "PANW", "CRWD", "NOW", "SNOW", "SHOP", "SQ"
]

def get_sp500_raw_data():
    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "subtype", "operation": "in_range", "right": ["common"]},
            {"left": "market_cap_basic", "operation": "greater", "right": 1500000000},
            {"left": "Value.Traded", "operation": "greater", "right": 30000000}
        ],
        "columns": [
            "name", "close", "open", "high", "low", "volume", "change", "Value.Traded",
            "relative_volume_10d_calc",
            "average_true_range_14",
            "Perf.1M",
            "Perf.3M",
            "Volatility.D",
            "VWAP"
        ],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, 400]
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.tradingview.com/"
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    data = response.json()
    rows = []
    for item in data.get("data", []):
        d = item["d"]
        rows.append({
            "ticker": d[0],
            "close": float(d[1]) if d[1] is not None else 0.0,
            "open": float(d[2]) if d[2] is not None else 0.0,
            "high": float(d[3]) if d[3] is not None else 0.0,
            "low": float(d[4]) if d[4] is not None else 0.0,
            "volume": float(d[5]) if d[5] is not None else 0.0,
            "change_%": float(d[6]) if d[6] is not None else 0.0,
            "value_traded": float(d[7]) if d[7] is not None else 0.0,
            "rvol": float(d[8]) if len(d) > 8 and d[8] is not None else 1.0,
            "atr": float(d[9]) if len(d) > 9 and d[9] is not None else 1.0,
            "perf_1m": float(d[10]) if len(d) > 10 and d[10] is not None else 0.0,
            "perf_3m": float(d[11]) if len(d) > 11 and d[11] is not None else 0.0,
            "volatility": float(d[12]) if len(d) > 12 and d[12] is not None else 2.0,
            "vwap": float(d[13]) if len(d) > 13 and d[13] is not None else 0.0
        })
    return pd.DataFrame(rows)

def get_yfinance_fallback_data():
    """TradingView çöktüğünde devreye giren 2. Hat (Failover Motoru)."""
    print("⚠️ UYARI: TradingView yanıt vermedi! 2. Hat (yfinance Fallback) devreye giriyor...")
    try:
        data = yf.download(BACKUP_WATCHLIST, period="15d", interval="1d", group_by='ticker', progress=False)
        rows = []
        for t in BACKUP_WATCHLIST:
            try:
                df_t = data[t].dropna()
                if len(df_t) < 5:
                    continue
                last_row = df_t.iloc[-1]
                prev_row = df_t.iloc[-2]
                
                close = float(last_row['Close'])
                open_p = float(last_row['Open'])
                high = float(last_row['High'])
                low = float(last_row['Low'])
                vol = float(last_row['Volume'])
                
                prev_close = float(prev_row['Close'])
                change = ((close - prev_close) / prev_close) * 100.0
                value_traded = close * vol
                
                avg_vol_10 = df_t['Volume'].tail(10).mean()
                rvol = (vol / avg_vol_10) if avg_vol_10 > 0 else 1.0
                atr = (df_t['High'] - df_t['Low']).tail(14).mean()
                vwap = (high + low + close) / 3.0 # Tipik Fiyat VWAP yaklaşımı
                
                rows.append({
                    "ticker": t,
                    "close": close, "open": open_p, "high": high, "low": low,
                    "volume": vol, "change_%": round(change, 2),
                    "value_traded": value_traded, "rvol": round(rvol, 2),
                    "atr": round(atr, 2), "perf_1m": 0.0, "perf_3m": 0.0,
                    "volatility": 2.0, "vwap": vwap
                })
            except Exception:
                continue
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Yedek hat hatası: {e}")
        return pd.DataFrame()

def fetch_all_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 1. Hat: TradingView taranıyor...")
    try:
        df = get_sp500_raw_data()
        if not df.empty and len(df) > 10:
            df['tarih'] = pd.Timestamp.now().normalize()
            return df
    except Exception as e:
        print(f"TradingView Hatası: {e}")

    # TradingView başarısız olursa 2. Hat (Yedek Motor)
    df_fallback = get_yfinance_fallback_data()
    if not df_fallback.empty:
        df_fallback['tarih'] = pd.Timestamp.now().normalize()
        return df_fallback

    return pd.DataFrame()
