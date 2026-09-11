import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')
GECMIS_DOSYA = "sp500_gecmis_veri.csv"

def gecmis_veriyi_yukle():
    if os.path.exists(GECMIS_DOSYA):
        try:
            df = pd.read_csv(GECMIS_DOSYA)
            if 'tarih' in df.columns:
                df['tarih'] = pd.to_datetime(df['tarih'])
            return df
        except: 
            return pd.DataFrame()
    return pd.DataFrame()

def calculate_shock_scores(df, df_gecmis, dynamic_thresholds=None, dynamic_weights=None):
    if df.empty: 
        return df

    if dynamic_thresholds is None:
        dynamic_thresholds = {"th_vol": 1.5, "th_range": 1.4, "th_flow": 2.0, "th_lambda": 1.0}
    if dynamic_weights is None:
        dynamic_weights = {"vol": 0.20, "range": 0.25, "flow": 0.45, "lambda": 0.10}

    scored_data = []

    for idx, row in df.iterrows():
        item = row.to_dict()
        
        close = float(item.get('close', 0.0))
        open_p = float(item.get('open', close))
        high = float(item.get('high', close))
        low = float(item.get('low', close))
        change = float(item.get('change_%', 0.0))
        value_traded = float(item.get('value_traded', 0.0))
        rvol = float(item.get('rvol', 1.0))
        atr = float(item.get('atr', 1.0))
        perf_1m = float(item.get('perf_1m', 0.0))
        perf_3m = float(item.get('perf_3m', 0.0))
        vwap = float(item.get('vwap', 0.0))

        # 1. VWAP KONTROLÜ (Gün içi kurumsal maliyet üstünde mi?)
        is_below_vwap = False
        if vwap > 0 and close < (vwap * 0.994):
            is_below_vwap = True

        # 2. Z-SKORLAR
        z_vol = round(min(max(float((rvol - 1.0) * 2.5), -2.0), 6.0), 2)
        today_range = high - low
        safe_atr = max(atr, 0.01)
        z_range = round(min(max(float(((today_range / safe_atr) - 1.0) * 2.5), -2.0), 6.0), 2)

        liquidity_damping = min(value_traded / 50000000.0, 1.0) if value_traded > 0 else 0.0
        raw_lambda = ((abs(change) / ((value_traded / 25000000.0) + 1e-9)) * liquidity_damping) if value_traded > 0 else 0.0
        z_lambda = round(min(float(np.log1p(raw_lambda) * 2.0), 5.0), 2)

        if today_range > 0:
            clv = ((close - low) - (high - close)) / today_range
            body_eff = (close - open_p) / today_range
        else:
            clv = 0.0
            body_eff = 0.0
        aggressor_flow = (max(clv, 0.0) * 0.55) + (max(body_eff, 0.0) * 0.45)
        z_flow = round(float(aggressor_flow * 4.0), 2)

        # 3. GİRİŞ MARJI (SWEET SPOT: %1.5 ile %4.5 arası)
        entry_bonus = 0.0
        if 1.5 <= change <= 4.5:
            entry_bonus = 6.0
            entry_status = "🎯 İDEAL GİRİŞ BÖLGESİ"
        elif change >= 6.5:
            entry_bonus = -8.0
            entry_status = "⚠️ GEÇ KALINDI (Tepeden Alım Riski)"
        else:
            entry_status = "NORMAL GİRİŞ"

        shock_count = 0
        if z_vol >= dynamic_thresholds.get('th_vol', 1.5): shock_count += 1
        if z_range >= dynamic_thresholds.get('th_range', 1.4): shock_count += 1
        if z_lambda >= dynamic_thresholds.get('th_lambda', 1.0): shock_count += 1
        if z_flow >= dynamic_thresholds.get('th_flow', 2.0): shock_count += 1

        concordance_multiplier = 1.0 + (shock_count * 0.25)
        is_fresh_shock = (perf_1m <= 18.0) and (perf_3m >= -15.0)
        is_downtrend_knife = (perf_3m < -25.0) and (z_vol < 2.5)

        item['z_vol'] = z_vol
        item['z_range'] = z_range
        item['z_lambda'] = z_lambda
        item['z_flow'] = z_flow
        item['shock_count'] = shock_count
        item['concordance_mult'] = concordance_multiplier
        item['entry_bonus'] = entry_bonus
        item['entry_status'] = entry_status
        item['is_fresh_shock'] = is_fresh_shock
        item['is_downtrend_knife'] = is_downtrend_knife
        item['is_below_vwap'] = is_below_vwap
        scored_data.append(item)

    res_df = pd.DataFrame(scored_data)
    if res_df.empty: 
        return res_df

    res_df['pct_vol'] = res_df['z_vol'].rank(pct=True) * 100.0
    res_df['pct_range'] = res_df['z_range'].rank(pct=True) * 100.0
    res_df['pct_lambda'] = res_df['z_lambda'].rank(pct=True) * 100.0
    res_df['pct_flow'] = res_df['z_flow'].rank(pct=True) * 100.0

    w_v = dynamic_weights.get('vol', 0.25)
    w_r = dynamic_weights.get('range', 0.25)
    w_f = dynamic_weights.get('flow', 0.35)
    w_l = dynamic_weights.get('lambda', 0.15)

    base_score = (
        res_df['pct_vol'] * w_v +
        res_df['pct_range'] * w_r +
        res_df['pct_flow'] * w_f +
        res_df['pct_lambda'] * w_l
    ) * (res_df['concordance_mult'] / 1.5)

    raw_confidence = base_score + res_df['entry_bonus']
    final_score = np.clip(np.round(raw_confidence, 1), 0.0, 99.5)

    res_df['shock_score'] = np.where(
        (res_df['change_%'] > 0) & (~res_df['is_downtrend_knife']) & (~res_df['is_below_vwap']),
        final_score,
        0.0
    )
    res_df['confidence_score'] = res_df['shock_score']

    # KASA DAĞILIMI (POSITION SIZING)
    def assign_allocation(row):
        score = row['shock_score']
        chg = row['change_%']
        if score >= 85.0 and chg <= 5.5:
            return "⭐⭐⭐⭐⭐", "Portföyün %15 - %20'si (Yüksek Kurumsal Güven)"
        elif score >= 75.0:
            return "⭐⭐⭐⭐", "Portföyün %8 - %12'si (Dengeli Pozisyon)"
        elif score >= 65.0:
            return "⭐⭐⭐", "Portföyün %3 - %5'i (Deneme / Küçük Kasa)"
        else:
            return "⭐", "İşlem Açma (Yetersiz Güven)"

    stars_alloc = [assign_allocation(r) for _, r in res_df.iterrows()]
    res_df['stars'] = [sa[0] for sa in stars_alloc]
    res_df['allocation'] = [sa[1] for sa in stars_alloc]

    drop_cols = ['pct_vol', 'pct_range', 'pct_lambda', 'pct_flow', 'concordance_mult', 'is_downtrend_knife', 'is_below_vwap', 'entry_bonus']
    res_df = res_df.drop(columns=[col for col in drop_cols if col in res_df.columns])

    return res_df.sort_values(by='shock_score', ascending=False).reset_index(drop=True)
