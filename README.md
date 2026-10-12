# 🎬 Endüstriyel NLP: BiLSTM Tabanlı Duygu Analizi (Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-RESTful%20API-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Durum-Aktif%20Geliştirme-brightgreen)

IMDb film yorumlarını **"Olumlu"** veya **"Olumsuz"** olarak sınıflandıran, sıfırdan eğitilmiş, uçtan uca (full-stack) tasarlanmış bir Doğal Dil İşleme (NLP) sistemi. Model, FastAPI ile bir REST API'ye dönüştürülmüş ve karanlık temalı bir web arayüzü ile desteklenmiştir. Proje, yalnızca bir model eğitmekle kalmaz; modelin zayıf yönünü (olumsuzluk) ölçen test setleri, çok tohumlu deneyler ve tekrarlanabilir bir raporlama hattı da içerir.

---

## 📑 İçindekiler

- [Proje Mimarisi ve Özellikleri](#-proje-mimarisi-ve-özellikleri)
- [Proje Yapısı](#-proje-yapısı)
- [Geliştirme Yolculuğu](#️-geliştirme-yolculuğu-adım-adım-inşa)
- [Performans ve Kanıtlar](#-performans-ve-kanıtlar)
- [Deney Sonuçları](#-deney-sonuçları)
- [Temel Bulgular](#-temel-bulgular)
- [Bilinen Sınırlar ve Zayıf Yönler](#-bilinen-sınırlar-ve-zayıf-yönler)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Kullanılan Teknolojiler](#️-kullanılan-teknolojiler)
- [Yol Haritası](#-yol-haritası)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🚀 Proje Mimarisi ve Özellikleri

- **Ortak Veri Boru Hattı:** Eğitim, değerlendirme ve API aynı temizleme modülünü (`src/temizleme.py`) kullanır. `v2` temizleme, `isn't` gibi kısaltmaları `is not` biçimine açar ve `!`/`?` işaretlerini ayrı token olarak korur. 10.000 kelimelik sözlük yalnızca eğitim kısmından kurulur, metinler 256 tokene sabitlenir.
- **Derin Öğrenme Modeli:** Çift yönlü LSTM (BiLSTM), `pack_padded_sequence` ile yalnızca gerçek token'ları okur (PAD'ler son hafızayı bozmaz). İsteğe bağlı olarak son gizli durum yerine dikkat (attention) ağırlıklı özet kullanılabilir. Ezberlemeye karşı %50 `Dropout` ve gradient clipping uygulanır.
- **Tekrarlanabilir Eğitim:** Eğitim verisinin %10'u doğrulamaya ayrılır, en iyi epoch doğrulamaya göre seçilir, IMDb test seti yalnızca en sonda bir kez kullanılır. Her deney birden fazla rastgele tohumla çalıştırılır ve ortalama ± standart sapma raporlanır.
- **Olumsuzluk Odaklı Değerlendirme:** El yazımı iki ayrı olumsuzluk test seti (60 ve 44 cümle) ve test setlerindeki sıfatlarla çakışmayan dengeli şablon eğitim verisi.
- **Production-Ready API:** Seçilen model (`yayin/`), FastAPI ile RESTful API olarak sunulur. Hangi modelin yayınlanacağı yalnızca doğrulama doğruluğuna göre seçilir (`src/yayinla.py`).
- **Kullanıcı Arayüzü:** API ile asenkron (Fetch API) haberleşen, karanlık temalı (dark mode) bir HTML/CSS/JS frontend.

## 🗂 Proje Yapısı

```
nlp-bilstm-sentiment-analysis/
│
├── nlp_api.py                # yayin/ klasöründeki modeli servis eden FastAPI uygulaması
├── index.html                # Karanlık temalı web arayüzü
├── requirements.txt          # Bağımlılıklar
├── deney.sh                  # Çok tohumlu deney çalıştırıcı
├── degerlendir_negation.py   # Canlı API yolunu olumsuzluk seti 1 ile ölçer
│
├── imdb_lstm_modeli.pth      # İlk sürüm (baseline) model ağırlıkları
├── imdb_sozluk.json          # İlk sürüm (baseline) sözlüğü
├── nlp_lstm_mimarisi.py      # İlk sürümün eğitim betiği (baseline'ı üretti)
├── nlp_gercek_sinav.py       # İlk sürümün 25.000 yorumluk test değerlendirmesi
├── nlp_huggingface_api.py    # Hugging Face entegrasyonu için başlangıç betiği
│
├── src/
│   ├── temizleme.py          # Ortak metin temizleme (v1 eski, v2 yeni)
│   ├── veri.py               # Sözlük oluşturma ve sabit uzunluklu kodlama
│   ├── model.py              # BiLSTM (+ isteğe bağlı paketleme ve attention)
│   ├── egitim.py             # Tekrarlanabilir eğitim betiği
│   ├── sablon_veri.py        # Dengeli olumsuzluk şablon verisi üreticisi
│   ├── degerlendir.py        # Doğruluk ve negation değerlendirme yardımcıları
│   ├── degerlendir_kayitli.py# Kayıtlı modelleri yeniden eğitmeden ölçer
│   ├── metrik.py             # Kesinlik, duyarlılık, F1, karışıklık matrisi
│   ├── rapor.py              # Tüm modelleri ölçüp README tablosunu üretir
│   ├── ozet.py               # Sonuç dosyalarını tohum ortalamasıyla özetler
│   ├── veri_analizi.py       # Sözlük ve kırpma kayıplarını ölçer
│   └── yayinla.py            # Doğrulamaya göre en iyi modeli yayin/'e kopyalar
│
├── tests/
│   ├── negation_seti.json    # Olumsuzluk test seti 1 (60 cümle)
│   └── negation_seti_2.json  # Olumsuzluk test seti 2 (44 cümle, farklı kalıplar)
│
├── yayin/                    # API'nin kullandığı model, sözlük ve meta dosyası
├── sonuclar/                 # Deney sonuçları (JSON) ve rapor.md
├── arge_gunlugu/             # İlk AR-GE sürecinin deneme betikleri
└── .gitignore
```

> Not: `arge_gunlugu/` içindeki betikler, projenin ilk geliştirme sürecinin aşamalarını/denemelerini temsil eder. Güncel kod `src/` altındadır.

## 🛤️ Geliştirme Yolculuğu (Adım Adım İnşa)

Bu proje, hazır bir kütüphane fonksiyonunun tek satırda çağrıldığı bir yapı değil; veri setinden donanım entegrasyonuna kadar her adımın modüler olarak tasarlandığı bir AR-GE günlüğüdür.

### İlk sürüm (baseline)

| # | Aşama | İlgili Dosya |
|---|-------|--------------|
| 1 | Donanım ve altyapı testi — GPU (CUDA) kapasitesi ve VRAM erişiminin doğrulanması | `arge_gunlugu/nlp_pytorch_test.py` |
| 2 | Veri boru hattı ve temizlik — ham IMDb yorumlarının Regex ile büyük/küçük harf, noktalama ve HTML etiketlerinden arındırılması | `arge_gunlugu/nlp_metin_temizleme.py` |
| 3 | Özel sözlük inşası — frekans analiziyle donanım dostu 10.000 kelimelik bir embedding sözlüğü oluşturulması | `imdb_sozluk.json` |
| 4 | Matris paketlemesi — `Dataset`/`DataLoader` ile farklı uzunluktaki metinlerin 256'lık sabit tensörlere (padding/truncation) dönüştürülmesi | `arge_gunlugu/nlp_tam_boru_hatti.py` |
| 5 | Model inşası ve eğitim — BiLSTM mimarisinin kurulması ve 25.000 yorum üzerinde eğitilerek ağırlıkların (`.pth`) kaydedilmesi | `nlp_lstm_mimarisi.py` |
| 6 | Büyük yüzleşme — modelin hiç görmediği 25.000 satırlık test setiyle sınanması | `nlp_gercek_sinav.py` |
| 7 | Ürünleştirme — modelin FastAPI ile web servisine dönüştürülmesi ve asenkron bir arayüzle dış dünyaya açılması | `nlp_api.py`, `index.html` |

### İyileştirme turu (ölç, değiştir, yeniden ölç)

| # | Aşama | İlgili Dosya |
|---|-------|--------------|
| 8 | Olumsuzluk test setleri ve baseline ölçümü | `tests/`, `degerlendir_negation.py` |
| 9 | Tekrarlanabilir eğitim, doğrulama ayrımı ve çok tohumlu deney altyapısı | `src/egitim.py`, `deney.sh` |
| 10 | Temizleme `v2`, padding düzeltmesi, şablon verisi ve attention deneyleri | `src/temizleme.py`, `src/model.py`, `src/sablon_veri.py` |
| 11 | F1, kesinlik, duyarlılık ve karışıklık matrisi raporu | `src/rapor.py`, `src/metrik.py` |
| 12 | Doğrulamaya göre seçilen modelin API'ye alınması | `src/yayinla.py`, `nlp_api.py` |

## 📊 Performans ve Kanıtlar

Projenin ilk sürümündeki model, eğitim sırasında hiç görmediği **25.000 satırlık IMDb test veri setinde** değerlendirilmiş ve **%83.70 test doğruluğu** elde etmiştir. (Bu sayı bir doğrulama değil, test sonucudur; yeniden ölçülerek doğrulanmıştır.) İyileştirme turunun en iyi modeli aynı test setinde **%86 civarına** ulaşmıştır. Ayrıntılar aşağıdaki tabloda ve [Temel Bulgular](#-temel-bulgular) bölümündedir.

Uzun ve karmaşık metinlerde (örn. profesyonel film eleştirileri) model **%98'in üzerinde** eminlik oranlarına ulaşabilmektedir. Ancak yüksek eminlik, doğruluk anlamına gelmez: ilk sürüm, olumsuzluk test cümlelerindeki yanlış tahminlerinin bir kısmını %90'ın üzerinde eminlikle yapmıştır.

<!-- SONUC_TABLOSU_BASLA -->
## 📈 Deney Sonuçları

Değerler, koşular (farklı rastgele tohumlar) arasında ortalama ± standart sapmadır. Kesinlik, duyarlılık ve F1 "Olumlu" sınıfı içindir. `baseline`, projenin ilk sürümündeki model (`imdb_lstm_modeli.pth`) olup tek koşudur. Bu tablo `python3 -m src.rapor --readme` ile otomatik üretilir.

### IMDb test seti (25.000 yorum)

| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |
|---|---|---|---|---|---|---|
| baseline | 1 | 83.7 ± 0.0 | 83.3 ± 0.0 | 84.3 ± 0.0 | 83.8 ± 0.0 | 83.7 ± 0.0 |
| bilstm_v1 | 3 | 83.4 ± 0.2 | 84.5 ± 1.6 | 81.8 ± 2.8 | 83.1 ± 0.6 | 83.4 ± 0.2 |
| bilstm_v2 | 5 | 84.4 ± 0.5 | 83.9 ± 1.5 | 85.2 ± 2.7 | 84.5 ± 0.8 | 84.4 ± 0.5 |
| bilstm_v2_pack | 5 | 84.8 ± 0.4 | 86.5 ± 0.2 | 82.5 ± 0.6 | 84.4 ± 0.4 | 84.8 ± 0.4 |
| bilstm_v2_pack_sablon | 5 | 85.1 ± 0.2 | 85.0 ± 1.1 | 85.2 ± 1.3 | 85.1 ± 0.2 | 85.1 ± 0.2 |
| bilstm_v2_pack_dikkat | 5 | 86.1 ± 0.5 | 85.1 ± 1.9 | 87.7 ± 2.6 | 86.3 ± 0.6 | 86.1 ± 0.5 |
| bilstm_v2_pack_sablon_dikkat | 5 | 86.2 ± 0.6 | 86.3 ± 2.1 | 86.1 ± 4.2 | 86.1 ± 1.1 | 86.2 ± 0.7 |

### Olumsuzluk test seti 1 (60 cümle)

| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |
|---|---|---|---|---|---|---|
| baseline | 1 | 40.0 ± 0.0 | 42.1 ± 0.0 | 53.3 ± 0.0 | 47.1 ± 0.0 | 38.9 ± 0.0 |
| bilstm_v1 | 3 | 57.8 ± 4.2 | 68.6 ± 18.6 | 46.7 ± 26.5 | 50.0 ± 10.9 | 54.4 ± 3.3 |
| bilstm_v2 | 5 | 57.3 ± 4.8 | 63.1 ± 10.6 | 41.3 ± 6.5 | 49.1 ± 2.8 | 55.9 ± 3.8 |
| bilstm_v2_pack | 5 | 56.0 ± 4.2 | 58.6 ± 6.1 | 41.3 ± 3.8 | 48.5 ± 4.4 | 55.0 ± 4.1 |
| bilstm_v2_pack_sablon | 5 | 67.7 ± 1.9 | 75.2 ± 6.1 | 54.0 ± 5.5 | 62.5 ± 1.6 | 66.9 ± 1.5 |
| bilstm_v2_pack_dikkat | 5 | 57.0 ± 2.2 | 61.2 ± 4.4 | 39.3 ± 2.8 | 47.8 ± 2.0 | 55.6 ± 1.9 |
| bilstm_v2_pack_sablon_dikkat | 5 | 74.3 ± 6.9 | 81.3 ± 7.4 | 63.3 ± 12.5 | 70.7 ± 9.1 | 73.9 ± 7.2 |

### Olumsuzluk test seti 2 (44 cümle, farklı kalıplar)

| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |
|---|---|---|---|---|---|---|
| baseline | 1 | 52.3 ± 0.0 | 52.6 ± 0.0 | 45.5 ± 0.0 | 48.8 ± 0.0 | 52.0 ± 0.0 |
| bilstm_v1 | 3 | 58.3 ± 3.5 | 64.5 ± 9.5 | 43.9 ± 13.1 | 50.7 ± 5.1 | 56.7 ± 2.2 |
| bilstm_v2 | 5 | 57.7 ± 5.9 | 63.5 ± 10.2 | 39.1 ± 2.5 | 48.2 ± 4.5 | 56.2 ± 5.4 |
| bilstm_v2_pack | 5 | 59.5 ± 3.0 | 64.7 ± 4.3 | 41.8 ± 5.0 | 50.7 ± 4.7 | 58.2 ± 3.3 |
| bilstm_v2_pack_sablon | 5 | 62.3 ± 4.1 | 68.5 ± 8.5 | 47.3 ± 2.5 | 55.7 ± 2.7 | 61.4 ± 3.7 |
| bilstm_v2_pack_dikkat | 5 | 61.8 ± 1.9 | 67.7 ± 3.8 | 45.5 ± 0.0 | 54.4 ± 1.2 | 60.8 ± 1.7 |
| bilstm_v2_pack_sablon_dikkat | 5 | 60.0 ± 3.0 | 64.0 ± 3.9 | 45.5 ± 4.5 | 53.1 ± 4.3 | 59.1 ± 3.3 |

### Karışıklık matrisleri (koşu başına ortalama)

TN: olumsuzu olumsuz bildi, FP: olumsuzu olumlu sandı, FN: olumluyu olumsuz sandı, TP: olumluyu olumlu bildi.

| Model | Test seti | TN | FP | FN | TP |
|---|---|---|---|---|---|
| baseline | IMDb test | 10390 | 2110 | 1966 | 10534 |
| baseline | Negation 1 | 8 | 22 | 14 | 16 |
| baseline | Negation 2 | 13 | 9 | 12 | 10 |
| bilstm_v1 | IMDb test | 10621 | 1879 | 2276 | 10224 |
| bilstm_v1 | Negation 1 | 21 | 9 | 16 | 14 |
| bilstm_v1 | Negation 2 | 16 | 6 | 12 | 10 |
| bilstm_v2 | IMDb test | 10448 | 2052 | 1855 | 10645 |
| bilstm_v2 | Negation 1 | 22 | 8 | 18 | 12 |
| bilstm_v2 | Negation 2 | 17 | 5 | 13 | 9 |
| bilstm_v2_pack | IMDb test | 10889 | 1611 | 2188 | 10312 |
| bilstm_v2_pack | Negation 1 | 21 | 9 | 18 | 12 |
| bilstm_v2_pack | Negation 2 | 17 | 5 | 13 | 9 |
| bilstm_v2_pack_sablon | IMDb test | 10620 | 1880 | 1856 | 10644 |
| bilstm_v2_pack_sablon | Negation 1 | 24 | 6 | 14 | 16 |
| bilstm_v2_pack_sablon | Negation 2 | 17 | 5 | 12 | 10 |
| bilstm_v2_pack_dikkat | IMDb test | 10572 | 1928 | 1541 | 10959 |
| bilstm_v2_pack_dikkat | Negation 1 | 22 | 8 | 18 | 12 |
| bilstm_v2_pack_dikkat | Negation 2 | 17 | 5 | 12 | 10 |
| bilstm_v2_pack_sablon_dikkat | IMDb test | 10776 | 1724 | 1732 | 10768 |
| bilstm_v2_pack_sablon_dikkat | Negation 1 | 26 | 4 | 11 | 19 |
| bilstm_v2_pack_sablon_dikkat | Negation 2 | 16 | 6 | 12 | 10 |
<!-- SONUC_TABLOSU_BITIS -->

## 🔎 Temel Bulgular

Aşağıdaki sonuçlar, her biri 5 farklı rastgele tohumla eğitilen modellerin ortalamasıdır (baseline tek koşudur).

| Değişiklik | IMDb test doğruluğu | Olumsuzluk seti 1 | Olumsuzluk seti 2 |
|------------|--------------------|--------------------|--------------------|
| Baseline (ilk sürüm) | %83.7 | %40.0 | %52.3 |
| `v2` temizleme | %84.4 | %57.3 | %57.7 |
| + padding düzeltmesi (`pack_padded_sequence`) | %84.8 | %56.0 | %59.5 |
| + olumsuzluk şablon verisi | %85.1 | %67.7 | %62.3 |
| + attention (şablonsuz, pack ile karşılaştırma) | %86.1 | %57.0 | %61.8 |
| + şablon + attention | %86.2 | %74.3 ± 6.9 | %60.0 |

1. **Temizleme ve mimari iyileştirmeleri IMDb doğruluğunu küçük ama tutarlı biçimde artırdı** (%83.7 → ~%86). Attention tek başına yaklaşık +1.2 puan kazandırdı.
2. **Olumsuzluk sorunu çözülmedi.** Olumsuzluk içeren cümlelerde doğruluk hâlâ şans seviyesine (%50) yakın. Baseline'ın düşük skoru büyük olasılıkla "Olumlu" yanlılığından geliyordu; yeniden eğitilen modeller ters yönde ("Olumsuz") yanlılık gösteriyor. Bu yüzden baseline'dan sonraki sıçrama, olumsuzluğun anlaşıldığını değil, yanlılığın yön değiştirdiğini gösteriyor olabilir.
3. **Padding düzeltmesi negation'ı iyileştirmedi.** "Kısa cümlede PAD'ler son hafızayı bozuyor" hipotezi önceden belirlenmiş kriterle test edildi ve reddedildi.
4. **Şablon verisi şablona benzeyen cümlelerde işe yaradı, farklı kalıplarda yazılmış cümlelere genellenmedi.** Seti 1'de +11.7 puan, farklı kalıplı Set 2'de +2.8 puan (tohum sapması içinde).
5. **Tek koşuya güvenilmez.** Aynı kodla farklı tohumlar, 60 cümlelik sette 10 puana varan fark üretti (en kötü örnek: aynı model ailesi için %15 ile %73 arasında "Olumlu" tahmin oranı). Bu yüzden tüm karşılaştırmalar çok tohumlu yapıldı.
6. **Hipotez (henüz doğrulanmadı):** Model, olumsuzluğun anlamını öğrenmek yerine istatistiksel kısayollar (örn. "not" geçiyorsa olumsuz) öğreniyor olabilir. Önceden eğitilmiş bir dil modeli (DistilBERT) bu soruya ayrı bir yanıt verecektir.

## 🛑 Bilinen Sınırlar ve Zayıf Yönler

Sistemin sınırlarını anlamak, gelişim sürecinin en kritik parçasıdır:

1. **Olumsuzluk Kapsamı (Negation Scope):** Olumsuzluk test setlerinde en iyi model bile şansın (%50) belirgin biçimde üstüne çıkamıyor. Özellikle "olumsuz sıfatı olumsuzlayıp olumluya varma" (*"not boring"*, *"hardly disappointing"*) cümlelerinde modeller büyük ölçüde başarısız. Şablon verisi ve attention bunu yalnızca şablona benzeyen cümlelerde iyileştirdi.
2. **Küçük Test Setleri:** İki olumsuzluk test setinde toplam 104 cümle var; tek cümle ~1.7-2.3 puana denk geliyor. İkinci set, karar vermek için birden fazla kez kullanıldığından artık tamamen bağımsız sayılmaz.
3. **Dil:** Model İngilizce IMDb verisiyle eğitildi ve yalnızca `a-z` karakterleri korunuyor. Türkçe girişler anlamlı sonuç vermez; Türkçe harfleri korumak tek başına bunu çözmez, çok dilli bir model gerekir.
4. **Kırpma:** Yorumlar 256 kelimeye kesiliyor. Eğitim yorumlarının %29'u bu sınırı aşıyor ve toplam kelimelerin %23.3'ü atılıyor. Sözlük dışı kelime oranı ise %5.8.
5. **Baseline Tek Koşu:** İlk sürüm tek bir eğitimle elde edildiği için kendi tohum varyansı bilinmiyor.

## 💻 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.10+
- (Opsiyonel ama önerilir) CUDA destekli bir GPU

### API'yi çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/hakanEfeTuysuz/nlp-bilstm-sentiment-analysis.git
cd nlp-bilstm-sentiment-analysis

# 2. Gerekli kütüphaneleri kurun
pip install -r requirements.txt

# 3. API sunucusunu başlatın
uvicorn nlp_api:app --reload
```

Ardından:
- Swagger UI üzerinden test etmek için tarayıcıda **http://127.0.0.1:8000/docs** adresine gidin.
- Ya da doğrudan `index.html` dosyasını tarayıcınızda açarak görsel arayüzü kullanın.

### Deneyleri yeniden üretme

```bash
# Tek bir model eğit (çıktı: modeller/ ve sonuclar/)
python3 -m src.egitim --ad bilstm_v2_pack --temizleme v2 --paketle --epoch 20

# Aynı ayarı 5 tohumla çalıştır
bash deney.sh bilstm_v2_pack_sablon v2 "1 2 3 4 5" --paketle --sablon 4000

# Tohum ortalamalarını özetle
python3 -m src.ozet bilstm_v2_pack

# Tüm modelleri ölç ve README tablosunu güncelle
python3 -m src.rapor bilstm_v --readme

# Doğrulamaya göre en iyi modeli API'ye al
python3 -m src.yayinla bilstm_v2_pack_sablon
```

İlk çalıştırmada IMDb veri seti indirilir; sonrasında `export HF_DATASETS_OFFLINE=1` ile çevrimdışı çalışabilirsiniz.

## 🛠️ Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Model / Eğitim | PyTorch (BiLSTM, attention, Dropout) |
| Veri | Hugging Face `datasets` (IMDb), Regex, özel tokenizer/sözlük |
| API | FastAPI, Uvicorn, Pydantic |
| Arayüz | HTML / CSS / JavaScript (Fetch API) |

## 🗺 Yol Haritası

Tamamlanan maddeler `[x]` ile işaretlidir.

### Aşama 1 — Değerlendirme Altyapısı
- [x] El yapımı olumsuzluk test setleri (iki set, farklı kalıplar)
- [x] Doğruluğun yanında kesinlik, duyarlılık, F1 ve karışıklık matrisi raporu
- [ ] Hata analizini hata türüne göre kategorize etmek (şimdilik hatalar kategori etiketiyle JSON'a kaydediliyor)
- [x] Sonuçları README'de tablolaştırmak (otomatik: `python3 -m src.rapor --readme`)
- [ ] Karar için kullanılmamış, üçüncü bir bağımsız olumsuzluk seti

### Aşama 2 — Metin Temizleme
- [x] `!` ve `?` işaretlerini ayrı token olarak tutmak
- [x] Kısaltmaları açmak (`isn't` → `is not`)
- [x] Sözlük dışı kelime ve kırpma oranlarını ölçmek
- [ ] Sözlük boyutu ve kırpma uzunluğunu denemek (örn. 400 token veya yorumun başı + sonu)
- [ ] Türkçe karakter desteği (bilerek ertelendi: model İngilizce eğitildi, çok dilli model gerektirir)

### Aşama 3 — Veri Tarafında Olumsuzluk
- [x] Dengeli şablon verisi (sınırlı genelleme: aynı kalıplara benzeyen cümlelerde işe yaradı)
- [ ] `not` sonrasındaki kelimelere `NOT_` öneki eklemek
- [ ] Farklı kalıplar içeren daha çeşitli olumsuzluk verisi

### Aşama 4 — Mimari
- [x] BiLSTM üzerine attention katmanı (IMDb'de +1.2 puan, olumsuzlukta tutarlı kazanç yok)
- [ ] Attention ağırlıklarını olumsuzluk cümlelerinde görselleştirmek
- [ ] Önceden eğitilmiş kelime gömmeleri (GloVe / fastText)
- [ ] DistilBERT fine-tune edip BiLSTM ile karşılaştırmak
- [ ] Çok dilli model (Türkçe girişler için)

### Aşama 5 — Yayınlama ve Mühendislik
- [x] `requirements.txt`
- [x] Eğitim, veri ve değerlendirme kodunun `src/` altında toplanması
- [ ] API'nin attention'lı modeli yüklemesi
- [ ] Modelin Hugging Face üzerinde yayınlanması (`nlp_huggingface_api.py` başlangıç noktası)
- [ ] API için birim testleri (`pytest`) ve olumsuzluk setini çalıştıran CI adımı
- [ ] Arayüzde güven oranını ve düşük güvenli tahminlerde uyarıyı göstermek
- [ ] Docker ile paketleme

## 🤝 Katkıda Bulunma

Katkılar, hata bildirimleri ve öneriler memnuniyetle karşılanır. Bir *issue* açabilir veya *pull request* gönderebilirsiniz.

## 📄 Lisans

Bu proje için henüz bir lisans belirlenmemiştir. Kullanmadan önce depo sahibiyle iletişime geçmeniz önerilir.