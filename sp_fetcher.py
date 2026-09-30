"""TradingView açık tarayıcısı (ücretsiz, anahtarsız) — bugünün kapanış barı yedeği + sektör bilgisi.

v2'de ana veri kaynağı yfinance panelidir (price_history.py). Bu modül yalnızca:
  * Yahoo bugünün barını henüz yayınlamadıysa panele eklenecek kesiti,
  * sektör etiketlerini (portföy sektör limiti için)
sağlar. Tickerlar Yahoo formatına çevrilir (BRK.B -> BRK-B).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests


def get_sp500_raw_data(limit: int = 700) -> pd.DataFrame:
    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "subtype", "operation": "in_range", "right": ["common"]},
            {"left": "market_cap_basic", "operation": "greater", "right": 5_000_000_000},
            {"left": "Value.Traded", "operation": "greater", "right": 20_000_000},
        ],
        "columns": ["name", "close", "open", "high", "low", "volume", "change", "Value.Traded", "sector"],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, limit],
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
               "Accept": "application/json", "Referer": "https://www.tradingview.com/"}
    res = requests.post(url, json=payload, headers=headers, timeout=20)
    rows = []
    for item in res.json().get("data", []):
        d = item.get("d", [])
        f = lambda i, dflt=0.0: float(d[i]) if len(d) > i and d[i] is not None else dflt
        rows.append({
            "ticker": str(d[0]).upper().replace(".", "-").replace("/", "-"),
            "close": f(1), "open": f(2), "high": f(3), "low": f(4), "volume": f(5),
            "change_%": f(6), "value_traded": f(7),
            "sector": d[8] if len(d) > 8 else None,
        })
    return pd.DataFrame(rows)


def fetch_all_data():
    """Geriye uyumluluk: TradingView kesitini döndürür (boşsa boş DataFrame)."""
    print(f"[{datetime.now():%H:%M:%S}] TradingView ABD kesiti alınıyor...")
    try:
        df = get_sp500_raw_data()
        if not df.empty:
            df["tarih"] = pd.Timestamp.now().normalize()
        return df
    except Exception as exc:
        print(f"TradingView hatası: {exc}")
        return pd.DataFrame()
