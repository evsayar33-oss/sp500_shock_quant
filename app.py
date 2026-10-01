"""Meta-Engine paneli — sade, mobil uyumlu, iki projede ortak (config ile yönetilir)."""
import importlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

import config as C

TITLE = getattr(C, "PROJECT_TITLE", "Meta-Engine")
ICON = getattr(C, "PROJECT_ICON", "📈")
CCY = getattr(C, "CCY", "")
st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px 16px; height: 100%;}
.card .k {font-size: .78rem; opacity: .65; margin-bottom: 2px;}
.card .v {font-size: 1.25rem; font-weight: 650; line-height: 1.3;}
.card .s {font-size: .78rem; opacity: .7; margin-top: 2px;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.ok {background: rgba(34,197,94,.15); color: #16a34a;}
.warn {background: rgba(234,179,8,.18); color: #b58900;}
.bad {background: rgba(239,68,68,.15); color: #dc2626;}
.muted {opacity: .65; font-size: .85rem;}
h1 {font-size: 1.6rem !important; margin-bottom: 0 !important;}
</style>
""", unsafe_allow_html=True)

REGIME_TXT = {"CRASH": ("Çöküş", "bad"), "STRESS": ("Stres", "warn"), "ROTATION": ("Rotasyon", "warn"),
              "EXPANSION": ("Yükseliş", "ok"), "QUIET": ("Sakin", "ok"), "NORMAL": ("Normal", "ok")}
MACRO_TXT = {"RISK_ON": ("Risk iştahı", "ok"), "NEUTRAL": ("Nötr", "ok"), "RISK_OFF": ("Riskten kaçış", "bad"),
             "TL_SHOCK": ("TL şoku", "bad"), "VOL_SHOCK": ("Volatilite şoku", "bad"), "UNKNOWN": ("Veri yok", "warn")}
GUARD_TXT = {"NORMAL": ("Normal", "ok"), "WATCH": ("Temkinli", "warn"), "SAFE": ("Koruma", "bad"),
             "RECOVERY": ("Toparlanma", "warn")}
FAMILY_TXT = {"event_score": "Olay şoku", "flow_score": "Birikim (CMF)", "activity_score": "Aktivite",
              "liquidity_score": "Likidite", "resilience_score": "Dayanıklılık", "sector_score": "Grup akışı"}


# ------------------------------------------------------------------
# Veri
# ------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=600)
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_prices(ticker, days=180):
    try:
        mod = importlib.import_module(getattr(C, "HISTORY_MODULE", "bist_history"))
        p = mod.load_panel(min_date=pd.Timestamp.now() - pd.Timedelta(days=days * 1.5))
        return p[p["ticker"] == ticker].sort_values("tarih").tail(days)
    except Exception:
        return pd.DataFrame()


state = load_json(C.AI_STATE_FILE)
report = load_json(C.BACKTEST_REPORT_FILE)
scan = load_csv(C.GECMIS_DOSYA)
ledger = load_csv(C.LEDGER_FILE)
if not scan.empty and "tarih" in scan.columns:
    scan["tarih"] = pd.to_datetime(scan["tarih"], errors="coerce")
    scan = scan.dropna(subset=["tarih"])
today = scan[scan["tarih"] == scan["tarih"].max()].copy() if not scan.empty else pd.DataFrame()
last = state.get("last_scan", {})
reg = last.get("regime", {})
guard = state.get("autonomy_guard", {})
sc = state.get("backtest_summary", {})
ml = state.get("meta_label", {})


def num(x, d=0.0):
    try:
        x = float(x)
        return d if not np.isfinite(x) else x
    except Exception:
        return d


def px(x):
    return f"{CCY}{num(x):,.2f}" if CCY == "$" else f"{num(x):,.2f} {CCY}"


def card(col, k, v, s="", cls=""):
    pill = f'<span class="pill {cls}">{v}</span>' if cls else v
    col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{pill}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def status_of(row):
    """(etiket, sınıf, nedenler) — hissenin bugünkü durumu ve sinyal olmama nedenleri."""
    reasons = []
    if bool(row.get("is_illiquid", False)):
        reasons.append("Likidite eşiğin altında")
    if bool(row.get("is_downtrend_knife", False)):
        reasons.append("Sert düşüş trendinde (düşen bıçak)")
    if num(row.get("overnight_risk"), 0) >= C.MAX_OVERNIGHT_RISK:
        reasons.append("Gece boşluğu (gap) riski yüksek")
    if num(row.get("flow_score"), 100) < C.MIN_FLOW_SCORE:
        reasons.append("Birikim (akış) zayıf")
    if bool(row.get("earnings_in_window", False)):
        reasons.append("Elde tutma süresinde bilanço var")
    if not bool(row.get("eligible", True)) and not reasons and num(row.get("change_%")) <= 0:
        reasons.append("Model bu rejimde yalnızca yükselen günde giriş yapıyor")
    score, thr = num(row.get("watch_score")), num(row.get("effective_min_score"), 75)
    if num(row.get("weight_pct")) > 0:
        return "SİNYAL", "ok", reasons
    if score < thr:
        reasons.append(f"Skor eşiğin altında ({score:.1f} / {thr:.1f})")
    pw = row.get("p_win")
    if ml.get("enabled") and pw == pw and pw is not None:
        q = num((ml.get("model") or {}).get("q"), 0.5)
        if num(pw) < q:
            reasons.append(f"Kazanma olasılığı filtreyi geçmedi (%{num(pw) * 100:.0f} < %{q * 100:.0f})")
    lfr = row.get("lf_reason")
    if isinstance(lfr, str) and lfr:
        reasons.append(f"Volatilite filtresi eledi: {lfr}")
    if not reasons:
        reasons.append("Portföy limitleri (grup / korelasyon / brüt) nedeniyle seçilmedi")
    near = score >= thr - 5
    return ("İZLEMEDE" if near else "UYGUN DEĞİL"), ("warn" if near else "bad"), reasons


# ------------------------------------------------------------------
# Başlık + özet kartları
# ------------------------------------------------------------------
st.title(f"{ICON} {TITLE}")
st.markdown(f'<div class="muted">Son tarama: <b>{last.get("day", "-")}</b> · Giriş: sonraki seans açılışı · '
            f'Çıkış: {C.HORIZON}. gün kapanışı</div>', unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
r_txt, r_cls = REGIME_TXT.get(reg.get("label"), (reg.get("label", "-"), "warn"))
m_txt, m_cls = MACRO_TXT.get(reg.get("macro_label"), (reg.get("macro_label", "-"), "warn"))
g_txt, g_cls = GUARD_TXT.get(guard.get("mode"), (guard.get("mode", "-"), "warn"))
card(c1, "Piyasa rejimi", r_txt, f"güven %{num(reg.get('confidence')) * 100:.0f}", r_cls)
card(c2, "Makro ortam", m_txt, f"stres {num(reg.get('macro_stress')):.2f}", m_cls)
card(c3, "Risk modu", g_txt, f"maruziyet %{num(reg.get('exposure_mult'), 1) * 100:.0f}", g_cls)
card(c4, "Model karnesi (test)", f"%{num(sc.get('win_rate')):.0f} kazanma",
     f"net %{num(sc.get('avg_return')):+.2f}/işlem · PF {num(sc.get('profit_factor')):.2f}"
     + (" · olasılık filtresi açık" if ml.get("enabled") else ""))
st.write("")

# ------------------------------------------------------------------
# Hisse arama
# ------------------------------------------------------------------
tickers = sorted(today["ticker"].dropna().unique().tolist()) if not today.empty else []
q = st.selectbox("🔍 Hisse ara", options=[""] + tickers, index=0, placeholder="Hisse kodu yazın…",
                 format_func=lambda t: "Hisse kodu yazın…" if t == "" else t, label_visibility="collapsed")
if q:
    row = today[today["ticker"] == q].iloc[0].to_dict()
    label, cls, reasons = status_of(row)
    a, b = st.columns([1.1, 1.4])
    with a:
        st.markdown(f"### {q} &nbsp; <span class='pill {cls}'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"**{px(row.get('close'))}** &nbsp; <span class='muted'>günlük %{num(row.get('change_%')):+.2f} · "
                    f"piyasaya göre %{num(row.get('excess_return')):+.2f}</span>", unsafe_allow_html=True)
        score, thr = num(row.get("watch_score")), num(row.get("effective_min_score"), 75)
        st.progress(min(max(score / 100.0, 0.0), 1.0), text=f"Skor {score:.1f} · eşik {thr:.1f}")
        k1, k2 = st.columns(2)
        pw = row.get("p_win")
        k1.metric("Kazanma olasılığı", f"%{num(pw) * 100:.0f}" if pw == pw and pw is not None else "—")
        k2.metric("Önerilen ağırlık", f"%{num(row.get('weight_pct')):.1f}")
        if row.get("grp_name"):
            st.markdown(f"🧭 **Grup:** {row['grp_name']}  \n<span class='muted'>grup birikimi (CMF) "
                        f"{num(row.get('sec_cmf')):+.2f} · 20g göreli %{num(row.get('sec_ret20')):+.1f}</span>",
                        unsafe_allow_html=True)
        if label != "SİNYAL":
            st.markdown("**Neden sinyal değil?**")
            for r_ in reasons:
                st.markdown(f"- {r_}")
    with b:
        fam = {FAMILY_TXT[k]: num(row.get(k), 50) for k in FAMILY_TXT if k in row}
        st.markdown("**Skor bileşenleri** <span class='muted'>(0–100, piyasadaki sırası)</span>", unsafe_allow_html=True)
        st.bar_chart(pd.Series(fam), height=210)
        pr = load_prices(q)
        if not pr.empty:
            st.markdown("**Fiyat (son 6 ay)**")
            st.line_chart(pr.set_index("tarih")["close"], height=200)
    st.divider()

# ------------------------------------------------------------------
# Sekmeler
# ------------------------------------------------------------------
t1, t6, t3, t2, t4, t7, t5 = st.tabs(["🎯 Sinyaller", "💰 Canlı K/Z", "📂 Pozisyonlar", "🧭 Para akışı", "📊 Backtest", "🩺 Sağlık", "📒 Defter"])

with t1:
    if today.empty:
        st.info("Henüz tarama verisi yok.")
    else:
        picks = today[pd.to_numeric(today.get("weight_pct", 0), errors="coerce").fillna(0) > 0]
        if picks.empty:
            st.info("Bugün yeni sinyal yok. Eşiğe en yakın hisseler:")
            watch = today[today.get("eligible", True) == True].sort_values("watch_score", ascending=False).head(8)  # noqa: E712
            view = watch[["ticker", "watch_score", "effective_min_score", "change_%"]].rename(
                columns={"ticker": "Hisse", "watch_score": "Skor", "effective_min_score": "Eşik", "change_%": "Günlük %"})
        else:
            cols = {"ticker": "Hisse", "close": "Fiyat", "change_%": "Günlük %", "watch_score": "Skor",
                    "effective_min_score": "Eşik", "p_win": "Kazanma olasılığı", "weight_pct": "Ağırlık %",
                    "grp_name": "Grup"}
            view = picks[[c for c in cols if c in picks.columns]].rename(columns=cols)
            if "Kazanma olasılığı" in view.columns:
                view["Kazanma olasılığı"] = (pd.to_numeric(view["Kazanma olasılığı"], errors="coerce") * 100).round(0)
        st.dataframe(view, hide_index=True, use_container_width=True,
                     column_config={"Skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
                                    "Günlük %": st.column_config.NumberColumn(format="%+.2f"),
                                    "Fiyat": st.column_config.NumberColumn(format="%.2f"),
                                    "Kazanma olasılığı": st.column_config.NumberColumn(format="%.0f%%")})

with t2:
    board = pd.DataFrame(load_json(os.path.join(C.DATA_DIR, "sector_board.json")) or [])
    st.caption("Gruplar, her ay birlikte hareket eden hisselerden istatistiksel olarak kurulur. "
               "Birikim = grubun ortalama para akışı (CMF) ve genişliği. 🕵️ = fiyat henüz hareket etmeden birikim.")
    if board.empty:
        st.info("Para akışı panosu bir sonraki taramada oluşacak.")
    else:
        for side, cls in (("BİRİKİM", "ok"), ("GÖRECE GÜÇLÜ", "warn"), ("DAĞITIM", "bad")):
            part = board[board["side"] == side]
            if part.empty:
                continue
            st.markdown(f"<span class='pill {cls}'>{side}</span>", unsafe_allow_html=True)
            for _, g in part.iterrows():
                spy = " 🕵️" if side == "BİRİKİM" and num(g.get("stealth")) >= 0.35 else ""
                st.markdown(f"**{g['name']}**{spy} &nbsp; <span class='muted'>{int(num(g['n']))} hisse · CMF {num(g['sec_cmf']):+.2f} · "
                            f"genişlik %{num(g['sec_acc']) * 100:.0f} · 20g %{num(g['sec_ret20']):+.1f}</span>",
                            unsafe_allow_html=True)

with t3:
    lt = load_csv(os.path.join(C.DATA_DIR, "live_trades.csv"))
    if lt.empty:
        st.info("Açık pozisyon yok.")
    else:
        op = lt[lt["Durum"].isin(["Açılışta alınacak", "Açık", "TP1 ✓ (stop girişte)"])]
        if op.empty:
            st.success("Açık / bekleyen pozisyon yok.")
        else:
            led = load_csv(C.LEDGER_FILE)
            lv = led[pd.to_numeric(led.get("label_version"), errors="coerce") == 3] if not led.empty else led
            lev = lv.set_index(["date", "ticker"])[["stop_price", "tp1_price", "tp2_price"]] if not lv.empty else None
            rows = []
            for _, r in op.iterrows():
                k = (r["Sinyal"], r["Hisse"])
                stp, tp1, tp2 = (lev.loc[k].tolist() if lev is not None and k in lev.index else [None, None, None])
                rows.append({"Hisse": r["Hisse"], "Sinyal": r["Sinyal"], "Durum": r["Durum"], "Giriş": r["Giriş"],
                             "K/Z %": r["Net %"], "Stop": stp, "TP1": tp1, "TP2": tp2, "Ağırlık %": r["Ağırlık %"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"K/Z %": st.column_config.NumberColumn(format="%+.2f"),
                                        "Giriş": st.column_config.NumberColumn(format="%.2f"),
                                        "Stop": st.column_config.NumberColumn(format="%.2f"),
                                        "TP1": st.column_config.NumberColumn(format="%.2f"),
                                        "TP2": st.column_config.NumberColumn(format="%.2f")})
            from report import rule_text
            st.caption("Çıkış kuralı: " + rule_text())

with t6:
    ls = load_json(os.path.join(C.DATA_DIR, "live_summary.json"))
    lt = load_csv(os.path.join(C.DATA_DIR, "live_trades.csv"))
    wk = load_csv(os.path.join(C.DATA_DIR, "live_weekly.csv"))
    la, l20, bt = ls.get("all") or {}, ls.get("last20") or {}, ls.get("backtest") or {}
    if not ls:
        st.info("Canlı takip bir sonraki taramada başlayacak.")
    else:
        a, b, c, d = st.columns(4)
        card(a, "Kapanan işlem", f"{la.get('n', 0)}", f"açık {ls.get('open', 0)} · bekleyen {ls.get('pending', 0)}")
        card(b, "Canlı kazanma", f"%{num(la.get('win_rate')):.0f}" if la.get("n") else "—",
             f"backtest beklentisi %{num(bt.get('win_rate')):.0f}")
        card(c, "İşlem başı net", f"%{num(la.get('avg_return')):+.2f}" if la.get("n") else "—",
             f"backtest %{num(bt.get('avg_return')):+.2f}")
        card(d, "TP1 / Stop oranı", f"%{num(ls.get('tp1_rate')):.0f} / %{num(ls.get('stop_rate')):.0f}" if la.get("n") else "—",
             f"son 20 kazanma %{num(l20.get('win_rate')):.0f}" if l20.get("n", 0) >= 20 else "son 20: veri birikiyor")
        st.write("")
        st.markdown(f"**{C.HORIZON} günlük işlem tablosu** <span class='muted'>(G1–G{C.HORIZON}: girişe göre günlük kapanış getirisi, %; "
                    f"canlı takip başlangıcı {ls.get('since', '-')})</span>", unsafe_allow_html=True)
        if lt.empty:
            st.info("Henüz sinyal yok.")
        else:
            gcols = [f"G{k}" for k in range(1, C.HORIZON + 1)]
            st.dataframe(lt, hide_index=True, use_container_width=True,
                         column_config={**{g: st.column_config.NumberColumn(format="%+.2f") for g in gcols},
                                        "Net %": st.column_config.NumberColumn(format="%+.2f"),
                                        "Giriş": st.column_config.NumberColumn(format="%.2f"),
                                        "Olasılık": st.column_config.NumberColumn(format="%.2f")})
        st.markdown("**Haftalık kazanç / zarar** <span class='muted'>(kapanış haftasına göre; portföy katkısı = Σ ağırlık × net)</span>",
                    unsafe_allow_html=True)
        if wk.empty:
            st.info("İlk işlemler kapandığında haftalık tablo oluşacak.")
        else:
            st.dataframe(wk, hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(format="%+.2f")
                                        for k in ("Ort. net %", "Portföy katkısı %", "Kümülatif katkı %")})
            st.markdown("**Kümülatif portföy katkısı (%)**")
            st.line_chart(wk.set_index("Hafta")["Kümülatif katkı %"], height=220)

with t7:
    hl = load_json(os.path.join(C.DATA_DIR, "health.json"))
    if not hl:
        st.info("Sağlık raporu bir sonraki taramada oluşacak.")
    else:
        cls = {"ok": "ok", "warn": "warn", "bad": "bad"}[hl.get("status", "warn")]
        st.markdown(f"### Sistem sağlığı <span class='pill {cls}'>{hl.get('score', 0)}/100</span>", unsafe_allow_html=True)
        st.caption(f"Son kontrol: {hl.get('ts', '-')}")
        items = pd.DataFrame(hl.get("items", []))
        if not items.empty:
            icon = {"ok": "🟢", "warn": "🟡", "bad": "🔴"}
            for grp, part in items.groupby("group", sort=False):
                st.markdown(f"**{grp}**")
                for _, it in part.iterrows():
                    st.markdown(f"{icon.get(it['level'], '')} **{it['name']}** — <span class='muted'>{it['detail']}</span>",
                                unsafe_allow_html=True)
        rt = state.get("retrain") or {}
        if rt.get("history"):
            with st.expander("Otomatik yeniden eğitim geçmişi"):
                st.dataframe(pd.DataFrame(rt["history"]).iloc[::-1], hide_index=True, use_container_width=True)

with t4:
    if not report:
        st.info("Performans raporu ilk denetimden sonra oluşur.")
    else:
        st.markdown(f"**Son karar:** {report.get('decision', '-')}")
        er = report.get("exit_rule") or {}
        if er:
            st.caption(f"Çıkış kuralı: stop {er.get('EXIT_STOP_ATR')}×ATR · TP1 {er.get('EXIT_TP1_ATR')}×ATR (%{num(er.get('EXIT_TP1_FRAC')) * 100:.0f} sat, "
                       f"stop girişe) · TP2 {er.get('EXIT_TP2_ATR')}×ATR · en geç {C.HORIZON}. gün · eğitim: {report.get('train_mode', '-')}")
        rows = {"Aktif model": report.get("active_oos", {}), "Aday model": report.get("candidate_oos", {}),
                "Olasılık filtreli": report.get("meta_oos", {})}
        tbl = pd.DataFrame({k: {"Kazanma %": v.get("win_rate"), "Güven alt sınırı %": v.get("wilson_lcb"),
                                "Net/işlem %": v.get("avg_return"), "Kâr faktörü": v.get("profit_factor"),
                                "Güven (t)": v.get("cohort_t"), "İşlem": v.get("n")} for k, v in rows.items() if v}).T
        st.dataframe(tbl, use_container_width=True)
        st.caption(f"Olasılık filtresi: {'AÇIK' if report.get('meta_enabled') else 'KAPALI'} — {report.get('meta_note', '')}")
        ex = report.get("exit_variants", {})
        if ex:
            st.markdown("**Çıkış kuralı: aktif kural vs yalnız süre çıkışı**")
            st.dataframe(pd.DataFrame({k: {"Kazanma %": v.get("win_rate"), "Net/işlem %": v.get("avg_return"),
                                           "Kâr faktörü": v.get("profit_factor")} for k, v in ex.items()}).T,
                         use_container_width=True)
        tr = load_csv(os.path.join(C.DATA_DIR, "oos_trades.csv.gz"))
        if not tr.empty:
            tr["tarih"] = pd.to_datetime(tr["tarih"])
            daily = tr.groupby("tarih")["net_ret"].mean() / 100.0 / C.HORIZON
            st.markdown("**Test özsermaye eğrisi** <span class='muted'>(eşit ağırlık, günlük kohortlar)</span>",
                        unsafe_allow_html=True)
            st.line_chart((1 + daily).cumprod(), height=240)
        folds = report.get("folds", [])
        if folds:
            with st.expander("Dilim ayrıntıları"):
                st.dataframe(pd.DataFrame([{"Test dönemi": f"{f['test_start']} → {f['test_end']}",
                                            "Aday kazanma %": f["candidate"]["win_rate"], "Aday net %": f["candidate"]["avg_return"],
                                            "Aktif net %": f["active"]["avg_return"], "İşlem": f["candidate"]["n"]}
                                           for f in folds]), hide_index=True, use_container_width=True)

with t5:
    if ledger.empty:
        st.info("Defter boş.")
    else:
        v3 = ledger[pd.to_numeric(ledger.get("label_version"), errors="coerce") == 3] if "label_version" in ledger.columns else ledger
        done = pd.to_numeric(v3.loc[v3.get("status") == "CLOSED", "net_ret"], errors="coerce").dropna() if "status" in v3.columns else pd.Series(dtype=float)
        if len(done):
            a, b, c = st.columns(3)
            a.metric("Kapanan işlem", len(done))
            b.metric("Kazanma", f"%{(done > 0).mean() * 100:.0f}")
            c.metric("Net/işlem", f"%{done.mean():+.2f}")
        st.dataframe(v3.sort_values("date", ascending=False), hide_index=True, use_container_width=True)
