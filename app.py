import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="S&P 500 Meta-Engine v2", layout="wide", page_icon="🗽")

AI_STATE_FILE = "sp500_ai_state.json"
GECMIS_DOSYA = "sp500_gecmis_veri.csv"
LEDGER_FILE = "backtest_ledger.csv"
REPORT_FILE = os.path.join("data", "backtest_report.json")


def _json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


state, report = _json(AI_STATE_FILE), _json(REPORT_FILE)
scan, ledger = _csv(GECMIS_DOSYA), _csv(LEDGER_FILE)
if not scan.empty and "tarih" in scan.columns:
    scan["tarih"] = pd.to_datetime(scan["tarih"], errors="coerce")

last = state.get("last_scan", {})
reg = last.get("regime", {})
guard = state.get("autonomy_guard", {})
bt = state.get("backtest_summary", {})

st.title("🗽 S&P 500 Adaptive Meta-Engine v2")
st.caption(f"{state.get('status', '')} | Son tarama: {last.get('day', '-')} | "
           f"Sinyal kapanışta, giriş ertesi gün açılışta, çıkış T+5 kapanışta")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🌐 Rejim", reg.get("label", "-"), f"kesitsel {reg.get('cross_label', '-')}")
c2.metric("🌍 Makro", reg.get("macro_label", "-"), f"stres {float(reg.get('macro_stress') or 0):.2f}")
c3.metric("🛡️ Otonomi", guard.get("mode", "-"), f"maruziyet x{float(reg.get('exposure_mult') or 1):.2f}")
c4.metric("📊 OOS Win-Rate (net)", f"%{bt.get('win_rate', 0):.1f}",
          f"LCB %{bt.get('wilson_lcb', 0):.1f} | N={bt.get('n', 0)}")

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Günün Sinyalleri", "🛡️ Pozisyonlar", "🧪 Walk-Forward Raporu", "📒 Defter"])

with tab1:
    if scan.empty:
        st.info("Henüz tarama verisi yok.")
    else:
        today = scan[scan["tarih"] == scan["tarih"].max()].copy()
        if "effective_min_score" not in today.columns:
            today["effective_min_score"] = 75.0
        picks = today[today["shock_score"] >= today["effective_min_score"]].sort_values("shock_score", ascending=False)
        cols = [c for c in ["ticker", "shock_score", "effective_min_score", "stars", "close", "change_%",
                            "excess_return", "z_vol", "cmf20", "event_score", "flow_score", "resilience_score",
                            "overnight_risk", "sector", "weight_pct", "allocation", "entry_status"] if c in picks.columns]
        if picks.empty:
            st.info("Bugün eşiği geçen aday yok.")
            watch = today.sort_values("watch_score", ascending=False).head(10)
            st.dataframe(watch[[c for c in ["ticker", "watch_score", "effective_min_score", "change_%"] if c in watch.columns]],
                         hide_index=True, use_container_width=True)
        else:
            st.dataframe(picks[cols], hide_index=True, use_container_width=True,
                         column_config={"shock_score": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.1f")})
        q = st.text_input("Hisse röntgeni (örn. NVDA)").upper().strip()
        if q:
            h = scan[scan["ticker"] == q].sort_values("tarih")
            if h.empty:
                st.warning("Kayıt yok.")
            else:
                st.line_chart(h.set_index("tarih")[[c for c in ["watch_score", "effective_min_score"] if c in h.columns]])
                st.json(h.iloc[-1].dropna().to_dict(), expanded=False)

with tab2:
    if ledger.empty or "label_version" not in ledger.columns:
        st.info("v2 defter kaydı henüz yok.")
    else:
        v2 = ledger[pd.to_numeric(ledger["label_version"], errors="coerce") == 2]
        open_pos = v2[pd.to_numeric(v2["is_completed"], errors="coerce").fillna(0) == 0]
        if open_pos.empty:
            st.success("Açık / bekleyen pozisyon yok.")
        else:
            st.dataframe(open_pos[[c for c in ["date", "ticker", "entry_date", "entry_price", "weight_pct", "atr",
                                               "return_d1", "return_d3", "initial_score", "regime"] if c in open_pos.columns]],
                         hide_index=True, use_container_width=True)

with tab3:
    if not report:
        st.info("Walk-forward raporu henüz üretilmedi (shock_auditor.py).")
    else:
        st.write(f"**Karar:** {report.get('decision')} | {report.get('days')} etiketli gün | T+{report.get('horizon')} net")
        rows = {k: report.get(k, {}) for k in ("active_oos", "candidate_oos", "template_oos")}
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
        st.subheader("OOS Bilgi Katsayısı (günlük Spearman)")
        st.dataframe(pd.DataFrame(report.get("ic_oos", {})).T, use_container_width=True)
        st.subheader("Dilimler")
        folds = report.get("folds", [])
        if folds:
            ft = pd.DataFrame([{"test": f"{f['test_start']} → {f['test_end']}",
                                "aday_WR": f["candidate"]["win_rate"], "aday_net": f["candidate"]["avg_return"],
                                "aktif_WR": f["active"]["win_rate"], "aktif_net": f["active"]["avg_return"],
                                "n": f["candidate"]["n"]} for f in folds])
            st.dataframe(ft, hide_index=True, use_container_width=True)
        tr = _csv(os.path.join("data", "oos_trades.csv.gz"))
        if not tr.empty:
            tr["tarih"] = pd.to_datetime(tr["tarih"])
            eq = tr.groupby("tarih")["net_ret"].mean().iloc[::5] / 100.0
            st.subheader("Örtüşmeyen kohort özsermaye eğrisi (OOS, eşit ağırlık)")
            st.line_chart((1 + eq).cumprod())

with tab4:
    if ledger.empty:
        st.info("Defter boş.")
    else:
        st.dataframe(ledger.sort_values("date", ascending=False), hide_index=True, use_container_width=True)
        if "net_ret_5d" in ledger.columns:
            done = pd.to_numeric(ledger["net_ret_5d"], errors="coerce").dropna()
            if len(done):
                st.metric("Canlı net win-rate (v2)", f"%{(done > 0).mean() * 100:.1f}", f"N={len(done)} | ort %{done.mean():+.2f}")
