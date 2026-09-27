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

`config.py` (tüm sabitler) · `price_history.py` (panel, bileşenler, makro, bilanço) · `features.py` · `regime.py` · `sp_engine.py` · `portfolio.py` · `sp_learner.py` · `win_rate_optimizer.py` · `sp_fetcher.py` · `main.py` · `sp_auditor.py` · `app.py`. `autonomy_guard.py` değişmedi.

## Bilinen sınırlar

- Endeksin geçmiş üyelik tarihleri ücretsiz ve güvenilir biçimde alınamaz. Evren bugünkü üyelerle başlar ve yalnızca genişler; bu yüzden ilk backtest bir miktar iyimser olabilir.
- Yahoo fiyatları temettüye göre düzeltilmez; temettü günlerinde küçük bir etiket gürültüsü oluşur.
- Bilanço geçmişi `get_earnings_dates` ile yaklaşık 4 yıl geriye alınır; eksik kalan hisselerde karartma uygulanamaz.
- Kredi spread'i için HYG−LQD vekili kullanılır (gerçek OAS verisi ücretli).
- Türkiye'den işlem yapılıyorsa aracı kurumun **kur makası ve komisyonu** `config.py` içindeki `COMMISSION_BPS` değerine eklenmelidir.
