import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Sayfa Konfigürasyonu (Mobil Uyumlu & Geniş Ekran)
st.set_page_config(
    page_title="S&P 500 Quant Momentum & Exit Terminal",
    layout="wide",
    page_icon="🗽"
)

# Özel CSS ile Modern Koyu Tema ve Kart Tasarımı
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2962FF;
        margin-bottom: 10px;
    }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

AI_STATE_FILE = "sp500_ai_state.json"
GECMIS_DOSYA = "sp500_gecmis_veri.csv"
LEDGER_FILE = "backtest_ledger.csv"

def load_ai_state():
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "thresholds": {"min_score": 75.0, "th_vol": 1.5, "th_flow": 2.0},
        "weights": {"vol": 0.25, "flow": 0.35, "range": 0.25, "lambda": 0.15},
        "status": "AKTİF / OTONOM ÖĞRENME DEVREDE"
    }

def load_data():
    df_scan = pd.DataFrame()
    df_ledger = pd.DataFrame()
    
    if os.path.exists(GECMIS_DOSYA):
        try:
            df_scan = pd.read_csv(GECMIS_DOSYA)
            if 'tarih' in df_scan.columns:
                df_scan['tarih'] = pd.to_datetime(df_scan['tarih'])
        except:
            pass

    if os.path.exists(LEDGER_FILE):
        try:
            df_ledger = pd.read_csv(LEDGER_FILE)
        except:
            pass

    return df_scan, df_ledger

# Verileri Yükle
ai_state = load_ai_state()
df_scan, df_ledger = load_data()

th = ai_state.get('thresholds', {})
w = ai_state.get('weights', {})
min_score = th.get('min_score', 75.0)

# BAŞLIK
st.title("🗽 S&P 500 Quant Momentum & Otonom Terminal")
st.caption(f"🤖 **Model Durumu:** {ai_state.get('status', 'AKTİF')} | 🎯 **Hedef Baraj:** {min_score:.1f} Puan")

# ÜST METRİK PANELİ
c1, c2, c3, c4 = st.columns(4)

# Backtest Tamamlanma Sayısı ve Kazanma Oranı
completed_trades = df_ledger[df_ledger['is_completed'] == 1] if not df_ledger.empty and 'is_completed' in df_ledger.columns else pd.DataFrame()
win_rate = 0.0
if not completed_trades.empty and len(completed_trades) > 0:
    wins = completed_trades[completed_trades['return_d5'] > 0]
    win_rate = (len(wins) / len(completed_trades)) * 100.0

c1.metric("🎯 Dinamik Baraj", f"{min_score:.1f} Puan", f"Flow Ağırlığı: %{int(w.get('flow', 0.35)*100)}")
c2.metric("🏆 5G Win Rate", f"%{win_rate:.1f}", f"{len(completed_trades)} Tamamlanmış İşlem")
c3.metric("🧪 Backtest İlerlemesi", f"{len(completed_trades)} / 25", "25'te Otomatik Simülasyon")
c4.metric("⚡ Hacim Ağırlığı", f"%{int(w.get('vol', 0.25)*100)}", "Sıfır Gecikmeli Mikroyapı")

st.divider()

# --- YAN MENÜ (SIDEBAR) HİSSE SORGULAMA ---
st.sidebar.header("🔍 Wall Street Hisse Röntgeni")
search_ticker = st.sidebar.text_input("Hisse Sembolü (Örn: NVDA, TSLA):").upper().strip()

if search_ticker and not df_scan.empty:
    h_data = df_scan[df_scan['ticker'] == search_ticker]
    if not h_data.empty:
        last_row = h_data.sort_values(by='tarih', ascending=False).iloc[0]
        st.sidebar.subheader(f"#{search_ticker} Analizi")
        st.sidebar.metric("Quant Güven Skoru", f"{last_row['shock_score']:.1f}", last_row.get('stars', '⭐⭐⭐⭐'))
        st.sidebar.write(f"**Son Fiyat:** ${last_row['close']:.2f} ({last_row['change_%']:+.2f}%)")
        st.sidebar.write(f"**Giriş Durumu:** {last_row.get('entry_status', 'NORMAL')}")
        st.sidebar.write(f"**Alıcı Akış Baskısı:** {last_row.get('z_flow', 0.0):+.2f}σ")
        st.sidebar.write(f"**Hacim Şoku:** {last_row.get('z_vol', 0.0):+.2f}σ")
        st.sidebar.info(f"💰 {last_row.get('allocation', 'Standart Risk')}")
    else:
        st.sidebar.warning("Hisse son tarama kayıtlarında bulunamadı.")

# --- SEKME DÜZENİ ---
tab1, tab2, tab3 = st.tabs(["🚀 Günün Giriş Liderleri", "🛡️ Açık Pozisyonlar & Çıkışlar", "🧪 Canlı Backtest Defteri"])

# =========================================================================
# TAB 1: GÜNÜN GİRİŞ LİDERLERİ
# =========================================================================
with tab1:
    st.subheader("🎯 Bugünün Yüksek Güvenli Kurumsal Girişleri")
    st.markdown("*Sabah açılışında hacim ve alıcı akışıyla kopan, ideal giriş bölgesindeki S&P 500 ve Mid-Cap hisseleri.*")

    if not df_scan.empty:
        son_tarih = df_scan['tarih'].max()
        df_today = df_scan[df_scan['tarih'] == son_tarih].copy()
        
        top_candidates = df_today[df_today['shock_score'] >= min_score].sort_values(by='shock_score', ascending=False)

        if not top_candidates.empty:
            disp_cols = ['ticker', 'shock_score', 'stars', 'close', 'change_%', 'entry_status', 'allocation', 'z_flow', 'z_vol']
            col_map = {
                'ticker': 'Hisse',
                'shock_score': 'Güven Skoru',
                'stars': 'Yıldız',
                'close': 'Fiyat ($)',
                'change_%': 'Günlük %',
                'entry_status': 'Bölge',
                'allocation': 'Önerilen Kasa',
                'z_flow': 'Akış (Z)',
                'z_vol': 'Hacim (Z)'
            }
            
            st.dataframe(
                top_candidates[disp_cols].rename(columns=col_map),
                column_config={
                    "Güven Skoru": st.column_config.ProgressColumn("Güven Skoru", min_value=0, max_value=100, format="%.1f"),
                    "Fiyat ($)": st.column_config.NumberColumn("Fiyat ($)", format="$%.2f"),
                    "Günlük %": st.column_config.NumberColumn("Günlük %", format="%+0.2f%%"),
                    "Akış (Z)": st.column_config.NumberColumn("Akış (Z)", format="%+.2fσ"),
                    "Hacim (Z)": st.column_config.NumberColumn("Hacim (Z)", format="%+.2fσ"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"ℹ️ Bugün {min_score:.1f} puan barajını aşan bir hisse tespit edilmedi. Piyasa dinleniyor.")
    else:
        st.info("Henüz tarama verisi bulunmuyor. Lütfen GitHub Actions üzerinden seansı başlatın.")

# =========================================================================
# TAB 2: AÇIK POZİSYONLAR & ÇIKIŞ / STOP MOTORU
# =========================================================================
with tab2:
    st.subheader("🛡️ Açık Pozisyonlar & Çıkış Alarmları (Exit Engine)")
    st.markdown("*Son 5 gün içinde girilmiş açık işlemlerin anlık kâr/zarar ve stop durumu.*")

    if not df_ledger.empty and 'is_completed' in df_ledger.columns:
        open_pos = df_ledger[df_ledger['is_completed'] == 0].copy()
        
        if not open_pos.empty:
            # Güncel fiyat eşleştirmesi
            if not df_scan.empty:
                son_tarih = df_scan['tarih'].max()
                current_map = dict(zip(df_scan[df_scan['tarih'] == son_tarih]['ticker'], df_scan[df_scan['tarih'] == son_tarih]['close']))
            else:
                current_map = {}

            pos_cards = []
            for idx, r in open_pos.iterrows():
                tk = r['ticker']
                curr_p = current_map.get(tk, r['entry_price'])
                entry_p = float(r['entry_price'])
                pnl = ((curr_p - entry_p) / entry_p) * 100.0

                action = "🟢 TAŞIMAYA DEVAM"
                action_color = "#26a69a"
                
                if pnl <= -3.0:
                    action = "🚨 STOP-LOSS / ÇIKIŞ YAP"
                    action_color = "#ef5350"
                elif pnl >= 6.5:
                    action = "💰 KÂR KİLİTLE (Yarısını Sat)"
                    action_color = "#ffd54f"

                pos_cards.append({
                    "Hisse": tk,
                    "Giriş Tarihi": r['date'],
                    "Giriş Fiyatı": f"${entry_p:.2f}",
                    "Güncel Fiyat": f"${curr_p:.2f}",
                    "Kâr/Zarar (%)": pnl,
                    "Aksiyon Sinyali": action
                })

            df_pos_show = pd.DataFrame(pos_cards)
            st.dataframe(
                df_pos_show,
                column_config={
                    "Kâr/Zarar (%)": st.column_config.NumberColumn("Kâr/Zarar (%)", format="%+0.2f%%")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✅ Şu an takip edilen açık pozisyon bulunmuyor (Tümü vadesinde kapatıldı).")
    else:
        st.info("Kayıt defterinde henüz açık pozisyon yok.")

# =========================================================================
# TAB 3: CANLI BACKTEST DEFTERİ
# =========================================================================
with tab3:
    st.subheader("🧪 Şeffaf Backtest Defteri & Model Karnesi")
    st.markdown("*Bugünden itibaren biriken ve 25 işleme ulaştığında otomatik Grid Search çalıştıracak canlı kayıt defteri.*")

    if not df_ledger.empty:
        st.dataframe(
            df_ledger.sort_values(by='date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Kayıt defteri ilk seans açılışında otomatik doldurulacaktır.")
