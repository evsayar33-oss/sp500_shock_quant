"""Telegram raporları — sade, mobilde okunaklı, iki projede ortak tasarım.

Tasarım ilkeleri
  * Önce karar: "bugün ne yapmalıyım?" ilk ekranda.
  * Her sinyal 3 satır: hisse/fiyat · kanıt (skor, olasılık, grup) · uygulama (ağırlık, stop).
  * Teknik terimler yerine Türkçe karşılıklar; sayılar hizalı <pre> bloklarında.
  * Model karnesi tek satır; ayrıntı panelde (app.py).
"""
from __future__ import annotations

import html
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

TITLE = getattr(C, "PROJECT_TITLE", "Meta-Engine")
ICON = getattr(C, "PROJECT_ICON", "📈")
CCY = getattr(C, "CCY", "")
SEP = "━━━━━━━━━━━━━━━━━━"

REGIME_TXT = {"CRASH": "🔴 Çöküş", "STRESS": "🟠 Stres", "ROTATION": "🔵 Rotasyon", "EXPANSION": "🟢 Yükseliş",
              "QUIET": "⚪ Sakin", "NORMAL": "⚪ Normal"}
MACRO_TXT = {"RISK_ON": "🟢 Risk iştahı", "NEUTRAL": "⚪ Nötr", "RISK_OFF": "🔴 Riskten kaçış",
             "TL_SHOCK": "🔴 TL şoku", "VOL_SHOCK": "🔴 Volatilite şoku", "UNKNOWN": "⚪ Veri yok"}
GUARD_TXT = {"NORMAL": "🟢 Normal", "WATCH": "🟡 Temkinli", "SAFE": "🔴 Koruma (yeni giriş yok)",
             "RECOVERY": "🟠 Toparlanma"}
TR_MONTHS = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]


def _e(x) -> str:
    """Telegram HTML için güvenli metin ('<', '>', '&' etiket sanılmasın). v3.1'de BIST mesajı
    't=0.81 < 1.0' yüzünden reddediliyor ve ham etiketlerle gönderiliyordu."""
    return html.escape(str(x), quote=False)


def _plain(msg: str) -> str:
    """HTML yine reddedilirse: etiketleri temizleyip düz metin gönder (ham etiket asla görünmez)."""
    return html.unescape(re.sub(r"</?(b|i|pre|code)>", "", msg))


def _d(day) -> str:
    d = pd.Timestamp(day)
    return f"{d.day} {TR_MONTHS[d.month - 1]} {d.year}"


def _f(x, default=0.0):
    try:
        x = float(x)
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def _px(x):
    x = _f(x)
    return f"{CCY}{x:,.2f}" if CCY == "$" else f"{x:,.2f}{CCY}"


def send(message: str):
    import requests
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("CHAT_ID")
    if not token or not chat_id:
        print(message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    parts, cur = [], ""
    for block in message.split("\n\n"):
        if len(cur) + len(block) + 2 > 3800 and cur.strip():
            parts.append(cur.strip())
            cur = ""
        cur += block + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    for msg in parts:
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            res = requests.post(url, json=payload, timeout=15)
            if not res.json().get("ok"):
                print(f"Telegram HTML reddetti: {res.json().get('description')}")
                payload.pop("parse_mode", None)
                payload["text"] = _plain(msg)
                requests.post(url, json=payload, timeout=15)
        except Exception as exc:
            print(f"Telegram hatası: {exc}")


# ------------------------------------------------------------------
# Günlük sinyal mesajı
# ------------------------------------------------------------------
def rule_text() -> str:
    s, a, b = getattr(C, "EXIT_STOP_ATR", 2), getattr(C, "EXIT_TP1_ATR", 1), getattr(C, "EXIT_TP2_ATR", None)
    if b:
        return f"TP1 {a:g}×ATR'de yarısı satılır, stop girişe · TP2 {b:g}×ATR · stop {s:g}×ATR · en geç {C.HORIZON}. gün"
    return f"TP1 {a:g}×ATR'de tamamı satılır · stop {s:g}×ATR · en geç {C.HORIZON}. gün"


EVENT_TXT = {"GİRİŞ": ("🔵", "girildi"),
             "TP1": ("✅", "TP1 geldi · kâr alındı · kapandı" if not getattr(C, "EXIT_TP2_ATR", None)
                     else "TP1 geldi · yarısı satıldı · stop girişe çekildi"),
             "TP2": ("🏁", "TP2 geldi · kapandı"), "STOP": ("⛔", "stop · kapandı"), "BAŞABAŞ": ("⚪", "başabaş stop · kapandı"),
             "SÜRE": ("⏰", f"{getattr(C, 'HORIZON', 5)}. gün · kapandı")}


def _health_line(h):
    if not h:
        return None
    return f"Sağlık   {h.get('icon', '')} {h.get('score', 0)}/100"


def _levels_line(p, approx=False):
    t2 = f" · TP2 {_px(p['tp2'])}" if p.get("tp2") else ""
    pre = "≈ " if approx else ""
    return f"   {pre}Stop {_px(p.get('stop'))} · TP1 {_px(p.get('tp1'))}{t2}"


def scan_message(ctx: dict) -> str:
    reg, guard = ctx.get("regime", {}), ctx.get("guard", {})
    picks, positions = ctx.get("picks", []), ctx.get("positions", [])
    hl = ctx.get("health") or {}
    lines = [f"{ICON} <b>{TITLE} · Günlük Sinyal</b>", f"<i>{_d(ctx['day'])} · {getattr(C, 'CLOSE_TEXT', 'kapanış')}</i>", ""]
    lines.append(f"Piyasa   {REGIME_TXT.get(reg.get('label'), reg.get('label'))}")
    lines.append(f"Makro    {MACRO_TXT.get(reg.get('macro_label'), reg.get('macro_label'))}")
    lines.append(f"Risk     {GUARD_TXT.get(guard.get('mode'), guard.get('mode'))} · maruziyet %{_f(ctx.get('exposure'), 1) * 100:.0f}")
    if hl:
        lines.append(_health_line(hl))
        bad = [i for i in hl.get("items", []) if i["level"] != "ok"][:3]
        for i in bad:
            lines.append(f"   {'🔴' if i['level'] == 'bad' else '🟡'} <i>{_e(i['name'])}: {_e(i['detail'])}</i>")
    lines += ["", SEP]

    events = ctx.get("events") or []
    if events:
        lines.append("📣 <b>BUGÜN</b>")
        for e in events:
            ic, txt = EVENT_TXT.get(e["event"], ("•", e["event"]))
            net = f" · net <b>%{_f(e['net']):+.2f}</b>" if e.get("net") is not None else ""
            lines.append(f"{ic} <b>{_e(e['ticker'])}</b> {txt}{net}")
        lines += ["", SEP]

    if picks:
        lines.append(f"🎯 <b>YENİ SİNYALLER ({len(picks)})</b>")
        lines.append(f"<i>Giriş: sonraki seans açılışı · {rule_text()}</i>")
        for i, p in enumerate(picks, 1):
            prob = f" · Kazanma olasılığı <b>%{_f(p.get('p_win')) * 100:.0f}</b>" if p.get("p_win") == p.get("p_win") and p.get("p_win") is not None else ""
            lines.append("")
            lines.append(f"<b>{i}) {_e(p['ticker'])}</b>  {_px(p.get('close'))}  <i>(%{_f(p.get('change')):+.1f})</i>")
            lines.append(f"   Skor {_f(p.get('score')):.0f}/{_f(p.get('thr')):.0f}{prob} · Ağırlık <b>%{_f(p.get('weight')):.1f}</b>")
            lines.append(_levels_line(p, approx=True))
            if p.get("group"):
                flow = "birikimde" if _f(p.get("sec_cmf")) > 0 else "görece güçlü"
                lines.append(f"   🧭 Grup {flow}: <i>{_e(p['group'])}</i>")
        lines.append("<i>≈ seviyeler kapanışa göre; giriş fiyatı kesinleşince güncellenir.</i>")
    else:
        lines.append("🎯 <b>Bugün yeni sinyal yok</b>")
        watch = ctx.get("watch", [])
        if watch:
            lines.append("<i>Eşiğe yakın izleme listesi:</i>")
            for w in watch:
                extra = ""
                if w.get("p_win") is not None and w.get("p_win") == w.get("p_win"):
                    extra = f" · olasılık %{_f(w['p_win']) * 100:.0f} (gereken %{_f(w.get('q')) * 100:.0f})"
                lines.append(f"• {_e(w['ticker'])}  skor {_f(w['score']):.1f}/{_f(w['thr']):.1f}{extra}")

    lf_out = ctx.get("lf_out") or []
    if lf_out:
        names = ", ".join(_e(x["ticker"]) for x in lf_out[:5]) + (" …" if len(lf_out) > 5 else "")
        lines.append(f"🧹 <i>Volatilite filtresi {len(lf_out)} sinyali eledi: {names}</i>")
    if positions:
        lines += ["", SEP, f"📂 <b>POZİSYONLAR ({len(positions)})</b>", "<pre>"]
        for p in positions:
            if p["status"] == "PENDING":
                lines.append(f"{p['ticker']:<7} açılışta al   %{_f(p.get('weight')):.1f}")
            else:
                tag = " TP1✓" if p["status"] == "TP1" else ""
                lines.append(f"{p['ticker']:<7} {p.get('held', 0)}/{C.HORIZON}g {_f(p.get('pnl')):+6.2f}%{tag}")
        lines.append("</pre>")
        for p in positions:
            if p["status"] != "PENDING":
                t2 = f" · TP2 {_px(p['tp2'])}" if p.get("tp2") == p.get("tp2") and p.get("tp2") else ""
                stop_txt = "stop giriş" if p["status"] == "TP1" else f"stop {_px(p.get('stop'))}"
                lines.append(f"<i>{_e(p['ticker'])}: {stop_txt} · TP1 {_px(p.get('tp1'))}{t2}</i>")

    live = ctx.get("live") or {}
    la = live.get("all") or {}
    if la.get("n"):
        l20 = live.get("last20") or {}
        lines += ["", SEP, f"💰 <b>CANLI SONUÇ</b> <i>({live.get('since', '')} sonrası)</i>",
                  f"{la['n']} kapanan işlem · kazanma <b>%{_f(la.get('win_rate')):.0f}</b> · işlem başı net <b>%{_f(la.get('avg_return')):+.2f}</b>"
                  + (f" · son 20: %{_f(l20.get('win_rate')):.0f}" if l20.get("n", 0) >= 20 else "")]
        wk = live.get("this_week")
        if ctx.get("week_end") and wk:
            lines.append(f"📅 Bu hafta: {int(_f(wk.get('İşlem'), 0))} işlem · kazanma %{_f(wk.get('Kazanma %')):.0f} · "
                         f"portföy katkısı <b>%{_f(wk.get('Portföy katkısı %')):+.2f}</b> · toplam %{_f(wk.get('Kümülatif katkı %')):+.2f}")
    elif live.get("n_signals") is not None:
        lines += ["", f"<i>💰 Canlı takip başladı: {live.get('n_signals', 0)} sinyal, henüz kapanan işlem yok.</i>"]

    board = ctx.get("board")
    if board is not None and not board.empty:
        up = board[board["side"] == "BİRİKİM"].head(3)
        if up.empty:
            up = board[board["side"] == "GÖRECE GÜÇLÜ"].head(2)
        dn = board[board["side"] == "DAĞITIM"].head(2)
        if len(up) or len(dn):
            lines += [SEP, "🧭 <b>PARA AKIŞI (gruplar)</b>"]
            for _, b in up.iterrows():
                stealth = " 🕵️" if _f(b.get("stealth")) >= 0.35 and b["side"] == "BİRİKİM" else ""
                dot = "🟢" if b["side"] == "BİRİKİM" else "🟡"
                lines.append(f"{dot} {_e(b['name'])}{stealth}")
            for _, b in dn.iterrows():
                lines.append(f"🔴 {_e(b['name'])}")

    sc = ctx.get("scorecard") or {}
    if sc.get("n"):
        meta = (" · olasılık filtresi açık" if ctx.get("meta_on") else "") + \
               (" · volatilite filtresi açık" if ctx.get("lf_on") else "")
        lines += ["", f"<i>📊 Backtest karnesi: kazanma %{_f(sc.get('win_rate')):.0f} · "
                      f"işlem başı net %{_f(sc.get('avg_return')):+.2f} · PF {_f(sc.get('profit_factor')):.2f}{meta}</i>"]
    return "\n".join(l for l in lines if l is not None)


# ------------------------------------------------------------------
# Denetim (öğrenme) mesajı
# ------------------------------------------------------------------
def _row(label, a, b, fmt):
    fa = fmt.format(a) if a is not None else "-"
    fb = fmt.format(b) if b is not None else "-"
    return f"{label:<12}{fa:>9}{fb:>9}"


def audit_message(ctx: dict) -> str:
    lines = [f"🧠 <b>{TITLE} · Model Denetimi</b>", f"<i>{_d(ctx.get('day', datetime.now()))} · "
             f"{ctx.get('days', 0)} gün, {ctx.get('tickers', 0)} hisse</i>", ""]
    if not ctx.get("ok"):
        lines.append(f"⏳ {_e(ctx.get('note', 'Isınma aşaması'))}")
        return "\n".join(lines)

    status = ctx.get("status")
    head = {"PROMOTED": "✅ <b>Yeni model devreye alındı</b>", "ROLLBACK": "↩️ <b>Önceki modele dönüldü</b>",
            "KEPT": "⏸️ <b>Mevcut model korundu</b>"}.get(status, status)
    lines += [head, f"<i>{_e(ctx.get('note', ''))}</i>"]
    rt = ctx.get("retrain") or {}
    if rt.get("strong"):
        lines.append(f"🔁 <i>Güçlü yeniden eğitim ({_e(', '.join(rt.get('reasons') or ['tetikleyici']))}) · seçilen: {_e(rt.get('mode'))}</i>")
    lines.append("")

    a, c = ctx["active"], ctx["candidate"]
    lines.append("<pre>")
    lines.append(f"{'Test (OOS)':<12}{'Aktif':>9}{'Aday':>9}")
    lines.append(_row("Kazanma %", a["win_rate"], c["win_rate"], "{:.1f}"))
    lines.append(_row("Net/işlem %", a["avg_return"], c["avg_return"], "{:+.2f}"))
    lines.append(_row("Kâr faktörü", a["profit_factor"], c["profit_factor"], "{:.2f}"))
    lines.append(_row("Güven (t)", a["cohort_t"], c["cohort_t"], "{:.1f}"))
    lines.append(_row("İşlem", a["n"], c["n"], "{:d}"))
    lines.append("</pre>")

    m = ctx.get("meta")
    if m:
        state = "🟢 AÇIK" if ctx.get("meta_on") else "⚪ KAPALI"
        lines.append(f"🎚️ <b>Kazanma olasılığı filtresi:</b> {state}")
        lines.append(f"<i>Filtreli: kazanma %{m['win_rate']:.1f} · net %{m['avg_return']:+.2f} · PF {m['profit_factor']:.2f} (n={m['n']})</i>")

    lf = ctx.get("lf") or {}
    if lf.get("filt") and lf.get("base"):
        b, f = lf["base"], lf["filt"]
        state = "🟢 AÇIK" if lf.get("on") else "⚪ KAPALI"
        lines.append(f"🧹 <b>Volatilite filtresi:</b> {state}")
        lines.append(f"<i>Aynı dönemde filtresiz: kazanma %{b['win_rate']:.1f} · net %{b['avg_return']:+.2f} · PF {b['profit_factor']:.2f}</i>")
        lines.append(f"<i>Filtreli: kazanma %{f['win_rate']:.1f} · net %{f['avg_return']:+.2f} · PF {f['profit_factor']:.2f} "
                     f"· kalan sinyal %{100 * float(lf.get('kept') or 0):.0f}</i>")
    elif lf.get("note"):
        lines.append(f"🧹 <i>{_e(lf['note'])}</i>")

    ex = ctx.get("exits") or {}
    rates = ex.get("_rates") or {}
    rows = {k: v for k, v in ex.items() if not k.startswith("_") and v.get("n")}
    if rows:
        lines += ["", f"🚪 <b>Çıkış kuralı</b> <i>({rule_text()})</i>", "<pre>", f"{'':<12}{'Kazanma':>8}{'Net':>9}"]
        for name, e in rows.items():
            lines.append(f"{name:<12}{e['win_rate']:>7.1f}%{e['avg_return']:>+8.2f}%")
        lines.append("</pre>")
        if rates:
            tp2 = f" · TP1 sonrası TP2 %{rates.get('tp2_given_tp1', 0):.0f}" if getattr(C, "EXIT_TP2_ATR", None) else ""
            lines.append(f"<i>🎯 TP1'e ulaşma <b>%{rates.get('tp1', 0):.0f}</b>{tp2} · stop %{rates.get('stop', 0):.0f}</i>")

    drivers = ctx.get("drivers") or []
    if drivers:
        gate = {"ANY": "yön şartı yok (zayıflıkta birikimi de yakalar)", "UP": "yalnızca yükselen gün"}.get(ctx.get("gate"), "")
        lines += ["", "🔬 <b>Neyi takip ediyor?</b>"] + ([f"<i>Giriş kapısı: {gate}</i>"] if gate else [])
        lines += [f"{'▲' if d['sign'] > 0 else '▼'} {d['name']}" for d in drivers]

    lines += ["", f"<i>Eşik ofseti {ctx.get('offset', 0):+.0f} · Risk modu {GUARD_TXT.get(ctx.get('guard'), ctx.get('guard'))}</i>"]
    live = ctx.get("live") or {}
    if live.get("n"):
        lines.append(f"<i>📒 Canlı defter: {live['n']} işlem · kazanma %{live['win_rate']:.0f} · net %{live['avg_return']:+.2f}</i>")
    return "\n".join(lines)


FAMILY_TXT = {"event": "Olay/hacim şoku", "flow": "Sessiz birikim (CMF)", "activity": "İşlem aktivitesi",
              "liquidity": "Likidite", "resilience": "Piyasaya göre dayanıklılık", "sector": "Grup/sektör para akışı"}


def drivers_from_profile(profile: dict, top: int = 4) -> list:
    """Profildeki en ağır aileler ve yönleri (▲ yüksek iyi, ▼ düşük iyi)."""
    w = (profile or {}).get("weights") or {}
    s = (profile or {}).get("signs") or {}
    items = sorted(w.items(), key=lambda kv: -kv[1])[:top]
    return [{"name": FAMILY_TXT.get(k, k) + (" (ters: yüksek olan kötü)" if s.get(k, 1) < 0 else ""),
             "sign": s.get(k, 1), "weight": v} for k, v in items]
