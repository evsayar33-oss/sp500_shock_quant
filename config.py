"""Merkezi konfigürasyon — S&P 500 Shock Quant v2.

Tüm modüller (canlı skor, backtest, öğrenme, denetim) AYNI sabitleri buradan okur.
Böylece eğitim-canlı tutarsızlığı (train/serve skew) yapısal olarak engellenir.

Tamamen ücretsiz veri kaynakları:
  * Yahoo Finance (yfinance)          -> günlük OHLCV geçmişi + makro seriler + bilanço tarihleri
  * Wikipedia S&P 500 bileşen listesi -> evren (yedek: TradingView büyük şirket taraması)
  * TradingView public scanner (JSON) -> bugünün barı yedeği
"""
from __future__ import annotations

import os

# ------------------------------------------------------------------
# Dosya yolları
# ------------------------------------------------------------------
DATA_DIR = "data"
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv")          # aylık parçalı panel: ohlcv_YYYY_MM.csv.gz
MACRO_FILE = os.path.join(DATA_DIR, "macro.csv.gz")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.json")
EARNINGS_FILE = os.path.join(DATA_DIR, "earnings_dates.csv")
BACKTEST_REPORT_FILE = os.path.join(DATA_DIR, "backtest_report.json")
OOS_TRADES_FILE = os.path.join(DATA_DIR, "oos_trades.csv.gz")

AI_STATE_FILE = "sp500_ai_state.json"
GECMIS_DOSYA = "sp500_gecmis_veri.csv"      # canlı skorlanmış günlük kesitler (app.py okur)
SIGNAL_LOG_FILE = "sp500_signals_log.csv"
LEDGER_FILE = "backtest_ledger.csv"

# ------------------------------------------------------------------
# Geçmiş veri
# ------------------------------------------------------------------
BACKFILL_YEARS = 3                  # ilk kurulumda indirilecek yıl
INCREMENTAL_PERIOD = "15d"          # günlük güncellemede çekilecek pencere
DOWNLOAD_CHUNK = 60                 # yfinance toplu indirme parça boyutu
MAX_UNIVERSE = 520                  # S&P 500 bileşenleri (+ sınıf hisseleri)
SPLIT_CHECK_TOL = 0.02              # örtüşen barda %2'den büyük fark => bölünme/düzeltme, tam yeniden indir
MAX_VALID_DAILY_MOVE = 0.45         # ABD'de marj yok; |getiri|>%45 => büyük olasılıkla düzeltilmemiş kurumsal işlem

TICKER_SUFFIX = ""                  # ABD hisseleri Yahoo'da sonek almaz (BRK.B -> BRK-B dönüşümü ayrıca yapılır)

MACRO_TICKERS = {
    "SPY": "SPY",        # piyasa trendi
    "RSP": "RSP",        # eşit ağırlıklı S&P 500 — genişlik (breadth) vekili
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",   # vade yapısı: VIX/VIX3M > 1 => backwardation (stres)
    "TNX": "^TNX",       # 10Y getiri
    "IRX": "^IRX",       # 13 haftalık T-bill
    "HYG": "HYG",        # yüksek getirili tahvil — kredi riski
    "LQD": "LQD",        # yatırım yapılabilir tahvil
    "DXY": "DX-Y.NYB",   # dolar endeksi
}

# ------------------------------------------------------------------
# İşlem tanımı (etiket = canlı işlem ile BİREBİR aynı)
# ------------------------------------------------------------------
HORIZON = 5                         # T+5 kapanışta çıkış
# Sinyal: T günü kapanış sonrası. Giriş: T+1 açılış. Çıkış: T+HORIZON kapanış.

# Maliyet modeli (tek yön, baz puan) — ABD büyük şirketleri
COMMISSION_BPS = 1.0                # komisyon + SEC/FINRA ücretleri (muhafazakâr)
SLIPPAGE_BASE_BPS = 2.0
SLIPPAGE_LIQ_COEF = 60.0            # slip_bps = base + coef / sqrt(günlük medyan işlem hacmi, milyon $)
SLIPPAGE_MAX_BPS = 25.0
FX_NOTE = "Türkiye'den işlem yapılıyorsa aracı kurum kur makası ayrıca eklenmelidir."

# ------------------------------------------------------------------
# Uygunluk filtreleri (canlı + backtest ortak)
# ------------------------------------------------------------------
MIN_LIQ_TL = 50_000_000.0           # 20g medyan dolar hacmi alt sınırı ($) (isim geriye uyum için korundu)
MIN_FLOW_SCORE = 45.0
MAX_OVERNIGHT_RISK = 82.0
TOP_K_PER_DAY = 8
EARNINGS_BLACKOUT = True            # elde tutma penceresinde bilanço varsa sinyal üretme (canlı + backtest)

# ------------------------------------------------------------------
# Özellik pencereleri
# ------------------------------------------------------------------
Z_WINDOW = 60                       # zaman serisi z-skor tabanı (hissenin kendi geçmişi)
Z_MIN_PERIODS = 20
ATR_WINDOW = 14
LIQ_WINDOW = 20
CMF_WINDOW = 20
GAP_WINDOW = 20
CORR_WINDOW = 60

# ------------------------------------------------------------------
# Öğrenme / walk-forward
# ------------------------------------------------------------------
WF_MIN_TRAIN_DAYS = 250
WF_TEST_DAYS = 63
WF_EMBARGO_DAYS = HORIZON + 1       # eğitim sonu ile test başı arası boşluk (etiket sızıntısı engeli)
THRESHOLD_GRID = [float(x) for x in range(60, 94, 2)]
MIN_TRADES_FOR_THRESHOLD = 60
PROMOTION_MIN_OOS_TRADES = 150
PROMOTION_MIN_LCB_LIFT = 1.0        # yüzde puan
PROMOTION_MIN_AVG_LIFT = 0.10       # yüzde puan (net getiri)
LIVE_ROLLBACK_MIN_TRADES = 40

# ------------------------------------------------------------------
# Portföy / pozisyon boyutu
# ------------------------------------------------------------------
RISK_PER_TRADE_PCT = 1.0            # pozisyon başına HORIZON-günlük 1σ risk bütçesi (% özsermaye)
MAX_POSITION_PCT = 12.0
MIN_POSITION_PCT = 1.0
MAX_POSITIONS = 8
MAX_PER_GROUP = 3                  # istatistiksel grup (küme) başına en fazla pozisyon
MAX_GROSS_PCT = 70.0
MAX_PAIR_CORR = 0.70
MAX_PER_SECTOR = 3

# Çıkış motoru. Test edilen strateji = T+HORIZON kapanışta ZAMAN çıkışıdır (etiketle birebir).
# ATR felaket stopu yalnızca kuyruk riskine karşı koruma; normal işleyişte nadiren tetiklenir.
STOP_ATR = 3.0                      # = EXIT_STOP_ATR (geriye uyum)
TP1_ATR = 2.0          # bilgi amaçlı hedef seviye (kısmi kâr opsiyonel, modelin parçası değil)

REGIMES = ("CRASH", "STRESS", "ROTATION", "EXPANSION", "QUIET", "NORMAL")
REGIME_SEVERITY = {"EXPANSION": 0, "QUIET": 1, "NORMAL": 2, "ROTATION": 3, "STRESS": 4, "CRASH": 5}


def slippage_bps(liq_tl):
    """Likiditeye bağlı kayma (tek yön, bps). Skaler veya numpy/pandas kabul eder."""
    import numpy as np
    liq_mn = np.maximum(np.asarray(liq_tl, dtype=float) / 1e6, 1.0)
    return np.clip(SLIPPAGE_BASE_BPS + SLIPPAGE_LIQ_COEF / np.sqrt(liq_mn), SLIPPAGE_BASE_BPS, SLIPPAGE_MAX_BPS)


def round_trip_cost_pct(liq_tl):
    """Gidiş-dönüş toplam maliyet, yüzde puan."""
    return 2.0 * (COMMISSION_BPS + slippage_bps(liq_tl)) / 100.0


# ------------------------------------------------------------------
# Sektör / grup akış katmanı (sector_flow.py)
# ------------------------------------------------------------------
CLUSTER_LOOKBACK = 250              # küme kurulumunda kullanılan geçmiş gün (piyasadan arındırılmış getiri)
CLUSTER_K = 16                      # en fazla küme sayısı
CLUSTER_MIN_OBS = 150
MIN_GROUP_SIZE = 4                  # bu sayıdan küçük gruplarda sektör özellikleri nötr (50)

# ------------------------------------------------------------------
# Sunum (Telegram + panel) ve terfi güveni
# ------------------------------------------------------------------
PROJECT_TITLE = "S&P 500 Meta-Engine"
PROJECT_ICON = "🗽"
CCY = "$"
CLOSE_TEXT = "NY kapanış"
PROMOTION_MIN_T = 1.0               # aday OOS kohort t-istatistiği en az bu olmalı (şansa bağlı terfiyi önler)
HISTORY_MODULE = "price_history"            # panelin fiyat grafiği için
META_MIN_T = 0.0                    # olasılık filtresi: yön şartı (asıl kanıt: LCB + dilim tutarlılığı)
META_MIN_FOLD_SHARE = 0.6           # filtre, dilimlerin en az %60ında aktif modeli net getiride geçmeli

# ------------------------------------------------------------------
# v3 çıkış kuralı (gerçek veride seçildi: ilk yarıda seçim, son yarıda sınama; kullanıcı onayı "Dengeli")
# ------------------------------------------------------------------
EXIT_STOP_ATR = 3.0                 # zarar kes: giriş − 3.0×ATR
EXIT_TP1_ATR = 1.5                  # TP1: giriş + 1.5×ATR -> yarısı satılır, stop girişe çekilir
EXIT_TP1_FRAC = 0.5
EXIT_TP2_ATR = 2.0                  # TP2: giriş + 2×ATR (TP1'i geçenlerin ~%54-58'i ulaşıyor)
ROLLING_TRAIN_DAYS = 375            # güçlü yeniden eğitimde denenecek yakın dönem penceresi (etiketli gün)
HEALTH_CLOSE_HOUR = 17                # yerel saat (workflow TZ); bu saatten sonra bugünün barı beklenir
