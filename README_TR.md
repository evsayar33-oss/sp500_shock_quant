# KUR-UNUT V1 — PROJECT-SPECIFIC WIN-RATE OPTIMIZATION

Bu paket dört projenin her birine **kendi operasyonel hedef ufkunda** win-rate odaklı, walk-forward doğrulamalı bir optimizer ekler.

## Temel ilke

Amaç `win rate = %100` gibi bir hedefi körlemesine kovalamak değildir. Her proje:

1. Geçmiş sonuçları toplar.
2. Mevcut score threshold çevresinde dar ve kontrollü adaylar dener.
3. Kronolojik walk-forward OOS testleri yapar.
4. **Primary objective = OOS win rate** kullanır.
5. Küçük örneklemi Wilson lower bound ile cezalandırır.
6. Gerçek getiri mevcutsa PF ve ortalama getiri bozulma korumaları uygular.
7. Yalnızca doğrulanmış iyileşmeyi `active_threshold` olarak promote eder.
8. Yeterli kanıt yoksa mevcut threshold'u değiştirmez.

Dolayısıyla sistemler kendi başlarına performanslarını iyileştirmeye çalışır; ancak aynı anda overfit riskini sınırlamaya devam eder.

## Proje hedefleri

- `bist_orderflow_quant`: T+3 win-rate
- `bist_shock_quant`: T+5 win-rate
- `sp500_shock_quant`: T+5 win-rate
- `us_smallcap_quant`: mature lifecycle WIN-rate (projenin mevcut 6 aylık operational metric'i)

## Kurulum

Her klasördeki dosyaları aynı repo köküne kopyalayın.

### BIST Orderflow

Yeni:
- `win_rate_optimizer.py`

Değişen:
- `longterm_auditor.py`
- `main.py`

### BIST Shock

Yeni:
- `win_rate_optimizer.py`

Değişen:
- `shock_auditor.py`
- `main.py`

### S&P 500 Shock

Yeni:
- `win_rate_optimizer.py`

Değişen:
- `sp_auditor.py`
- `main.py`

### US Small-Cap

Yeni:
- `win_rate_optimizer.py`

Değişen:
- `longterm_auditor.py`
- `main.py`

Mevcut `app.py`, UI veya tarihsel CSV/JSON dosyalarına bu paket içinde dokunulmaz.

## Çalışma mantığı

Optimizer her audit döngüsünde çalışır. Eğer OOS sonuçları mevcut threshold'a göre anlamlı biçimde daha yüksek win-rate göstermezse hiçbir şey değiştirmez.

Promosyon için temel korumalar:

- minimum örneklem
- +2.0 yüzde puanı ham OOS win-rate artışı
- +1.5 yüzde puanı Wilson lower-bound artışı
- varsa PF'nin %10'dan fazla bozulmaması
- varsa ortalama getirinin 0.25 yüzde puanından fazla bozulmaması

## Test

Repo kökünde ilgili proje için:

```bash
python -m py_compile win_rate_optimizer.py main.py <audit_file>.py
python -c "from win_rate_optimizer import wilson_lower_bound; print(wilson_lower_bound(45, 60))"
```

Bu optimizer mevcut sistemi değiştirmeden önce yalnızca `state["win_rate_optimizer"]` içine aday/karar bilgisi yazar.

## Önemli

Bu katman gelecekteki win-rate'i garanti etmez. Görevi, mevcut proje için **doğrulanmış** win-rate iyileştirmelerini otomatik olarak bulmak ve güvenli koşullarda uygulamaktır.
