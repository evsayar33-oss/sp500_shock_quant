"""Telegram raporları — sade, mobilde okunaklı, iki projede ortak tasarım.

Tasarım ilkeleri
  * Önce karar: "bugün ne yapmalıyım?" ilk ekranda.
  * Her sinyal 3 satır: hisse/fiyat · kanıt (skor, olasılık, grup) · uygulama (ağırlık, stop).
  * Teknik terimler yerine Türkçe karşılıklar; sayılar hizalı <pre> bloklarında.
  * Model karnesi tek satır; ayrıntı panelde (app.py).
"""
from __future__ import annotations

import os
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
                payload.pop("parse_mode", None)
                requests.post(url, json=payload, timeout=15)
        except Exception as exc:
            print(f"Telegram hatası: {exc}")


# ------------------------------------------------------------------
# Günlük sinyal mesajı
# ------------------------------------------------------------------
def scan_message(ctx: dict) -> str:
    reg, guard = ctx.get("regime", {}), ctx.get("guard", {})
    picks, positions = ctx.get("picks", []), ctx.get("positions", [])
    lines = [f"{ICON} <b>{TITLE} · Günlük Sinyal</b>", f"<i>{_d(ctx['day'])} · {getattr(C, 'CLOSE_TEXT', 'kapanış')}</i>", ""]

    lines.append(f"Piyasa   {REGIME_TXT.get(reg.get('label'), reg.get('label'))}")
    lines.append(f"Makro    {MACRO_TXT.get(reg.get('macro_label'), reg.get('macro_label'))}")
    lines.append(f"Risk     {GUARD_TXT.get(guard.get('mode'), guard.get('mode'))} · maruziyet %{_f(ctx.get('exposure'), 1) * 100:.0f}")
    if _f(ctx.get("dq"), 100) < 80:
        lines.append(f"⚠️ <i>Veri kalitesi düşük ({_f(ctx.get('dq')):.0f}/100) — sinyalleri temkinli değerlendirin</i>")
    lines += ["", SEP]

    if picks:
        lines.append(f"🎯 <b>YENİ SİNYALLER ({len(picks)})</b>")
        lines.append(f"<i>Giriş: sonraki seans açılışı · Çıkış: {C.HORIZON}. gün kapanışı</i>")
        for i, p in enumerate(picks, 1):
            prob = f" · Kazanma olasılığı <b>%{_f(p.get('p_win')) * 100:.0f}</b>" if p.get("p_win") == p.get("p_win") and p.get("p_win") is not None else ""
            lines.append("")
            lines.append(f"<b>{i}) {p['ticker']}</b>  {_px(p.get('close'))}  <i>(%{_f(p.get('change')):+.1f})</i>")
            lines.append(f"   Skor {_f(p.get('score')):.0f}/{_f(p.get('thr')):.0f}{prob}")
            lines.append(f"   Ağırlık <b>%{_f(p.get('weight')):.1f}</b> · Koruma stopu {_px(p.get('stop'))}")
            if p.get("group"):
                flow = "birikimde" if _f(p.get("sec_cmf")) > 0 else "görece güçlü"
                lines.append(f"   🧭 Grup {flow}: <i>{p['group']}</i>")
    else:
        lines.append("🎯 <b>Bugün yeni sinyal yok</b>")
        watch = ctx.get("watch", [])
        if watch:
            lines.append("<i>Eşiğe yakın izleme listesi:</i>")
            for w in watch:
                extra = ""
                if w.get("p_win") is not None and w.get("p_win") == w.get("p_win"):
                    extra = f" · olasılık %{_f(w['p_win']) * 100:.0f} (gereken %{_f(w.get('q')) * 100:.0f})"
                lines.append(f"• {w['ticker']}  skor {_f(w['score']):.1f}/{_f(w['thr']):.1f}{extra}")

    if positions:
        lines += ["", SEP, f"📂 <b>POZİSYONLAR ({len(positions)})</b>", "<pre>"]
        for p in positions:
            if p["status"] == "PENDING":
                lines.append(f"{p['ticker']:<7} açılışta al  %{_f(p.get('weight')):.1f}")
            else:
                flag = {"STOP": " ⛔ STOP", "EXIT": " ⏰ ÇIK", "HOLD": ""}.get(p["status"], "")
                lines.append(f"{p['ticker']:<7} {p['held']}/{C.HORIZON}g  {_f(p['pnl']):+6.2f}%{flag}")
        lines.append("</pre>")

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
                lines.append(f"{dot} {b['name']}{stealth}")
            for _, b in dn.iterrows():
                lines.append(f"🔴 {b['name']}")

    sc = ctx.get("scorecard") or {}
    if sc.get("n"):
        meta = " · olasılık filtresi açık" if ctx.get("meta_on") else ""
        lines += ["", f"<i>📊 Model karnesi (geçmiş test): kazanma %{_f(sc.get('win_rate')):.0f} · "
                      f"işlem başı net %{_f(sc.get('avg_return')):+.2f} · PF {_f(sc.get('profit_factor')):.2f}{meta}</i>"]
    if any(p.get("status") == "STOP" for p in positions):
        lines.append("<i>⛔ = felaket stopu kırıldı, sonraki açılışta çık</i>")
    return "\n".join(lines)


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
        lines.append(f"⏳ {ctx.get('note', 'Isınma aşaması')}")
        return "\n".join(lines)

    status = ctx.get("status")
    head = {"PROMOTED": "✅ <b>Yeni model devreye alındı</b>", "ROLLBACK": "↩️ <b>Önceki modele dönüldü</b>",
            "KEPT": "⏸️ <b>Mevcut model korundu</b>"}.get(status, status)
    lines += [head, f"<i>{ctx.get('note', '')}</i>", ""]

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

    ex = ctx.get("exits") or {}
    if ex:
        lines += ["", "🚪 <b>Çıkış stratejisi karşılaştırması</b>", "<pre>", f"{'':<11}{'Kazanma':>8}{'Net':>9}"]
        for name, e in ex.items():
            lines.append(f"{name:<11}{e['win_rate']:>7.1f}%{e['avg_return']:>+8.2f}%")
        lines.append("</pre><i>Kâr-al hedefi kazanma oranını yükseltir ama işlem başı neti düşürebilir; model net getiriyi korur.</i>")

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
