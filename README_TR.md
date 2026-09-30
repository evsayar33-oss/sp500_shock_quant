# S&P 500 Shock Quant v2: Adaptive Meta-Engine

Sistem tamamen ücretsizdir. GitHub Actions üzerinde çalışır ve veriyi Yahoo Finance (yfinance), Wikipedia'daki S&P 500 bileşen listesi ve TradingView'in açık tarayıcısından alır. Ücretli API ya da API anahtarı gerekmez; bilgisayara da ihtiyaç yoktur.

## Neden yeniden yazıldı?

v1'in canlı defterinde 420 tamamlanmış işlem vardı. Win rate **%39.3**, işlem başına ortalama getiri **−%1.32** idi (maliyet düşülmeden). Aşağıdaki tablo bu sonucun nedenlerini ve v2'deki çözümleri gösteriyor.

| # | v1 sorunu | v2 çözümü |
|---|---|---|
| 1 | Öğrenme (ham z × 15) ile canlı skor (yüzdelik × uyum) farklı formüllerdi; eşik, seçildiği verinin kendisinde test ediliyordu | Canlı skor, backtest ve öğrenme **tek `score_frame`** fonksiyonunu kullanır. Walk-forward **embargo'lu**; eşik yalnızca eğitim diliminde seçilir. |
| 2 | Tarama NY açılışından 40 dk sonra yarım bar üzerinde yapılıyordu; "kapanış denetimi" seans ortasında çalışıyordu | Tarama **NY kapanışından sonra** yapılır. Giriş **sonraki seansın açılışında**, çıkış **5. işlem gününün kapanışında**. Etiket de tam olarak bu işlemi ölçer. |
| 3 | T+5 takvim günüyle sayılıyordu (hafta sonu dahil) | Etiketler paneldeki **kesin işlem günlerinden** hesaplanır. |
| 4 | Günde 40–80 sinyal üretiliyordu; aynı sektörden kümeler oluşuyordu | Günde en fazla 8 aday. Portföyde en fazla 8 pozisyon, **sektör başına 3**. **Korelasyon filtresi** ρ>0.70 olan ikinci hisseyi eler. |
| 5 | "S&P 500" adı altında 1.5 milyar $ üstü tüm ABD hisseleri taranıyordu | Evren **gerçek S&P 500 bileşenleri** (Wikipedia, ücretsiz). Endeksten çıkan hisselerin geçmişi korunur (survivorship bias azaltılır), ama işlem adayı yalnızca güncel üyelerdir. |
| 6 | z-skorlar sabit katsayılı dönüşümlerdi; "flow" tek bir mumdan hesaplanıyordu | **Gerçek zaman-serisi z-skoru** (her hisse kendi 60 gününe göre). Akış = **Chaikin Money Flow + mum baskısı** (OHLCV birikim vekili). Overnight riski gerçekleşmiş gap volatilitesinden ölçülür. |
| 7 | Sabit −%2.5 stop ve +%9 hedef volatiliteyi yok sayıyordu ve hiç test edilmemişti | Test edilen kural **T+5 zaman çıkışıdır**. 2.5×ATR felaket stopu yalnızca kuyruk riskine karşıdır. |
| 8 | "Kelly" fonksiyonu hiç çağrılmıyordu; ağırlık skora bağlı sabit bir metindi | **Volatilite hedefli** ağırlık: her pozisyon 5 günde yaklaşık %1 risk taşır. Buna brüt limit ve rejim/otonomi çarpanları uygulanır. |
| 9 | Rejim ve makro katmanı yoktu (koruma katmanı "UNKNOWN" görüyordu) | Kesitsel rejim + **ABD makro katmanı**: VIX yüzdeliği ve **VIX/VIX3M vade yapısı**, SPY/200g trendi, **HYG−LQD kredi**, RSP−SPY genişlik, 10Y faiz şoku, DXY. VOL_SHOCK veya RISK_OFF durumunda rejim yükselir, eşik artar, maruziyet azalır. |
| 10 | Maliyet yoktu; bilanço kontrolü yalnızca canlıda yapılıyordu | Likiditeye bağlı maliyet etiketten düşülür. **Bilanço karartması** hem backtest'te hem canlıda aynıdır: [T, T+5] içinde bilanço açıklaması varsa sinyal yok. yfinance takvimi ek kontrol olarak kalır. |

Ek olarak:
- IC, günlük kesitsel Spearman olarak ölçülür.
- Terfi yalnızca OOS kanıtla olur: Wilson LCB, net ortalama ve PF korumaları.
- Canlı defter bozulursa önceki stabil profile rollback yapılır.
- Win-rate optimizer global eşik ofsetini ayarlar.

## Günlük akış

- **Hafta içi UTC 21:30** (TSİ 00:30, yaz saatinde NY 17:30): `main.py`, ardından `sp_auditor.py` çalışır ve Telegram'a iki mesaj gelir.
- **Cumartesi**: S&P 500 bileşen listesi, 3 yıllık panel, makro seriler ve bilanço tarihleri baştan indirilir, ardından model yeniden denetlenir.

## Kurulum (yalnızca Android telefonla)

1. Repoya bu paketteki dosyaları yükle (**Add file → Upload files**). GitHub zip dosyasını açmaz, bu yüzden dosyaları tek tek seç.
2. `.github/workflows/daily_scan.yml` ve `daily_audit.yml` dosyalarını aç, kalem simgesiyle düzenle ve yeni içeriği yapıştır.
3. **Silinecek dosyalar:** `apply_guard_patch.py` ve `VERIFY_RESULTS.txt`. Eski `sp_engine.py`, `sp_fetcher.py` ve `sp_auditor.py` yenileriyle değiştirilir; aynı isimli dosyaları yükleyince üzerine yazılır.
4. **Dokunma:** `backtest_ledger.csv` ve `sp500_ai_state.json`. v1 ayarları otomatik olarak `archive_v1` altına taşınır. v1 defter satırları "legacy" olarak kalır; v2 kayıtları `label_version=2` ile işaretlenir.
5. **Actions** sekmesinde önce **"SP500 v2 Haftalik Tam Veri Yenileme" → Run workflow** çalıştır (ilk sefer ≈10–25 dk). Sonra **"SP500 v2 NY Kapanis Taramasi + Ogrenme" → Run workflow** çalıştır. Bundan sonrası otomatik.

## Dosyalar

`config.py` (tüm sabitler) · `sector_flow.py` (grup akış katmanı) · `meta_label.py` (olasılık filtresi) · `report.py` (Telegram) · `price_history.py` (panel, bileşenler, makro, bilanço) · `features.py` · `regime.py` · `sp_engine.py` · `portfolio.py` · `sp_learner.py` · `win_rate_optimizer.py` · `sp_fetcher.py` · `main.py` · `sp_auditor.py` · `app.py`. `autonomy_guard.py` değişmedi.

## Bilinen sınırlar

- Endeksin geçmiş üyelik tarihleri ücretsiz ve güvenilir biçimde alınamaz. Evren bugünkü üyelerle başlar ve yalnızca genişler; bu yüzden ilk backtest bir miktar iyimser olabilir.
- Yahoo fiyatları temettüye göre düzeltilmez; temettü günlerinde küçük bir etiket gürültüsü oluşur.
- Bilanço geçmişi `get_earnings_dates` ile yaklaşık 4 yıl geriye alınır; eksik kalan hisselerde karartma uygulanamaz.
- Kredi spread'i için HYG−LQD vekili kullanılır (gerçek OAS verisi ücretli).
- Türkiye'den işlem yapılıyorsa aracı kurumun **kur makası ve komisyonu** `config.py` içindeki `COMMISSION_BPS` değerine eklenmelidir.


## v2.1: Sektör / Grup Akış Katmanı ve İşaretli Öğrenme

**Amaç:** Kurumsal para tek hisseye değil, birlikte hareket eden sepetlere (tema, faktör, endeks ağırlığı) girer. Sistem artık tek hisse şokunu, o hissenin grubunda birikim olup olmadığıyla birlikte değerlendiriyor.

**Nasıl çalışır (`sector_flow.py`):**
- Her ayın ilk işlem gününde, önceki 250 günün **piyasadan arındırılmış getiri korelasyonlarıyla** hisseler en fazla 16 gruba kümelenir. Kümeleme yalnızca o tarihten önceki veriyi kullanır (ileriye bakma yok) ve takvime sabittir; böylece canlı ile backtest aynı grupları üretir.
- Grup özellikleri: grup CMF'si (birikim), birikim genişliği, 20 günlük göreli güç, lider–takipçi farkı, hissenin gruptan tek başına sapması. Bunlar birleşerek 6. aile olan `sector_score`'u oluşturur.
- Resmi sektör etiketi yerine istatistiksel küme kullanılmasının nedeni gerçek veride görüldü: S&P'de aynı birikim sinyali kümelerle IC +0.043, resmi sektörle +0.007 verdi.

**İşaretli öğrenme:** Öğrenici artık bir ailenin işaretini çevirebilir. Bunun için eğitim verisinde t ≤ −2 gerekir. Gerçek BIST verisinde "olay" ve "aktivite" (gürültülü hacim şokları) T+5'te **geri dönüş** öngörüyor (t≈−5). Yani büyük oyuncunun izi yüksek sesli hacim patlaması değil, **sessiz birikim**.

**Gerçek veriyle walk-forward (OOS, Eki 2024 – Eyl 2026, embargo'lu, net):**

| | Önce | Sonra |
|---|---|---|
| BIST (öğrenen aday, terfi etti) | −%0.64 / PF 0.84 / WR %44.3 | **+%0.40 / PF 1.14 / WR %48.5** (8 dilimin 7'sinde daha iyi; t=1.35, henüz %5 anlamlılıkta değil) |
| S&P (sektörlü şablon) | +%0.25 / PF 1.13 / t=1.0 | **+%0.34 / PF 1.18 / t=3.6** |

**Risk:** Sektör akışı adayları aynı gruba yığabileceği için portföyde **grup başına en fazla 3 pozisyon** sınırı vardır (S&P'de ayrıca resmi sektör başına 3).

**Raporlar:** Telegram'da günlük "Sektör / Grup Akış Panosu" yer alır: birikim, görece güçlü ve dağıtım grupları, 🕵️ sessiz birikim işareti. Her önerinin grubu ve sektör akış skoru da gösterilir. Panelde yeni "🧭 Sektör Akışı" sekmesi var.

**Not:** Sektör ailesinin başlangıç şablon ağırlığı (%15) ve iç formülü, iki projenin tüm verisine bakılarak seçildi. Bu hafif bir veri gözetleme (data snooping) riski taşır. Bu yüzden ağırlığın gerçek değerini, embargo'lu walk-forward ve terfi kuralları üzerinden sistemin kendisi belirler.


## v2.2: Win-rate düzeltmeleri, olasılık filtresi, yeni panel ve Telegram

**Bulunan ve düzeltilen hatalar**
1. **Eşik ofseti çift sayılıyordu (win-rate optimizer).** Optimizer havuzu, mevcut ofset zaten gömülü halde normalize ediliyordu; ofset her çalıştırmada kendi üzerine ekleniyordu. Havuz artık ofsetten arındırılmış eşiğe göre normalize ediliyor.
2. **Optimizer günlük işlem sınırını uygulamıyordu.** Eşik testleri artık canlıdaki gibi günlük en fazla `TOP_K_PER_DAY` işlemle yapılıyor.
3. **Güven istatistiği (t) rastgele sapıyordu.** "Her 5. işlem günü" alt örneklemesi başlangıç gününe bağlıydı (net +%0.44 iken t = −0.27 çıkabiliyordu). Artık tüm günlük kohortlar kullanılıyor ve örtüşme düzeltmesi yapılıyor.
4. **Şansa bağlı terfi mümkündü.** BIST v2.1 t≈0 ile terfi etmişti. Artık aday modelin OOS t değeri en az `PROMOTION_MIN_T` (1.0) olmalı.
5. **Hisse tespitinde sabit "yalnızca yükselen günde al" kapısı vardı.** Gerçek BIST verisinde bu kapı kenarı bozuyordu: zayıf günde birikim yapan hisselerde net +%0.51 ölçüldü. Kapı (`UP` / `ANY`) artık her walk-forward diliminde öğreniliyor.

**Win rate'i yükselten ekleme: kazanma olasılığı filtresi (`meta_label.py`)**
- Kurumsal meta-etiketleme yöntemi: L2-cezalı lojistik model, her aday için P(net kazanç) tahmin eder. Yalnızca eğitimde seçilen p* eşiğini geçen adaylar alınır ve olasılığa göre sıralanır.
- Model **tüm uygun hisselerle** eğitilir. Gerçek veride dar havuz WR %45.8 verirken geniş havuz WR %56.8 verdi.
- **Kendiliğinden açılır/kapanır:** OOS'ta win-rate LCB'yi yükseltiyor ve net getiriyi bozmuyorsa açılır.
  - S&P (OOS): WR %50.6 → **%56.8**, net +%0.34 → **+%1.41**, PF 1.18 → **1.78** → **AÇIK**
  - BIST (OOS): net getiriyi düşürüyor → **KAPALI**

**Çıkış stratejisi karşılaştırması (yalnızca rapor):** Kâr-al (TP) hedefleri win rate'i %59–60'a çıkarıyor, ama işlem başı neti düşürüyor. Bu yüzden canlı kural T+5 zaman çıkışı olarak kalıyor. Karşılaştırma her denetimde raporlanır.

**Panel (`app.py`):** Sade kart tasarımı ve 🔍 **hisse arama çubuğu**. Seçilen hisse için durum (SİNYAL / İZLEMEDE / UYGUN DEĞİL), skor ve eşik, kazanma olasılığı, grup, 6 bileşenin dökümü, 6 aylık fiyat grafiği ve **"neden sinyal değil"** açıklaması gösterilir. Sekmeler: Sinyaller · Para akışı · Pozisyonlar · Performans · Defter.

**Telegram (`report.py`):** Her iki mesaj yeniden tasarlandı. Günlük sinyal mesajı önce kararı verir; her sinyal üç satırdır (fiyat · skor ve olasılık · ağırlık ve stop · grup). Denetim mesajında aktif ve aday model hizalı bir tabloda karşılaştırılır.
