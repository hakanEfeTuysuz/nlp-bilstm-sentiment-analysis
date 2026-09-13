# 🎬 Endüstriyel NLP: BiLSTM Tabanlı Duygu Analizi (Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-RESTful%20API-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Durum-Aktif%20Geliştirme-brightgreen)

IMDb film yorumlarını **"Olumlu"** veya **"Olumsuz"** olarak sınıflandıran, sıfırdan eğitilmiş, uçtan uca (full-stack) tasarlanmış bir Doğal Dil İşleme (NLP) sistemi. Model, FastAPI ile bir REST API'ye dönüştürülmüş ve karanlık temalı bir web arayüzü ile desteklenmiştir.

---

## 📑 İçindekiler

- [Proje Mimarisi ve Özellikleri](#-proje-mimarisi-ve-özellikleri)
- [Proje Yapısı](#-proje-yapısı)
- [Geliştirme Yolculuğu](#️-geliştirme-yolculuğu-adım-adım-inşa)
- [Performans ve Kanıtlar](#-performans-ve-kanıtlar)
- [Bilinen Sınırlar ve Zayıf Yönler](#-bilinen-sınırlar-ve-zayıf-yönler)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Kullanılan Teknolojiler](#️-kullanılan-teknolojiler)
- [Yol Haritası](#-yol-haritası)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🚀 Proje Mimarisi ve Özellikleri

- **Özel Veri Boru Hattı:** Ham metinler Regex ile temizlenmiş, kelime frekans analizi yapılarak donanım dostu 10.000 kelimelik özel bir sözlük (`imdb_sozluk.json`) inşa edilmiştir.
- **Donanım Optimizasyonu:** PyTorch `Dataset` ve `DataLoader` sınıfları yazılarak metinler 256 uzunluğunda tensörlere dönüştürülmüş, CUDA (GPU) üzerinde darboğazsız bir eğitim sağlanmıştır.
- **Derin Öğrenme Modeli:** Cümle bağlamını hem ileri hem geri okuyabilen Çift Yönlü LSTM (BiLSTM) mimarisi kullanılmıştır. Ezberlemeyi (overfitting) önlemek için %50 oranında `Dropout` katmanı sisteme entegre edilmiştir.
- **Production-Ready API:** Eğitilen model (`.pth`), FastAPI kullanılarak yüksek performanslı bir RESTful API'ye dönüştürülmüştür.
- **Kullanıcı Arayüzü:** API ile asenkron (Fetch API) haberleşen, karanlık temalı (dark mode) bir HTML/CSS/JS frontend yazılmıştır.

## 🗂 Proje Yapısı

```
nlp-bilstm-sentiment-analysis/
│
├── imdb_sozluk.json          # Frekans analiziyle oluşturulan 10.000 kelimelik özel sözlük
├── imdb_lstm_modeli.pth      # Eğitilmiş BiLSTM modelinin ağırlıkları
├── index.html                # Karanlık temalı web arayüzü
├── nlp_api.py                # Modeli servis eden FastAPI uygulaması
│
├── nlp_pytorch_test.py       # GPU/CUDA ve VRAM erişim testi
├── nlp_metin_temizleme.py    # Regex tabanlı metin temizleme adımı
├── nlp_sozluk_temel.py       # Sözlük (vocabulary) oluşturmanın ilk sürümü
├── nlp_guvenli_sozluk.py     # Sözlük oluşturma - güvenlik/doğrulama iyileştirmeleri
├── nlp_profesyonel_sozluk.py # Sözlük oluşturma - son/olgun sürüm
├── nlp_embedding.py          # Kelime gömme (embedding) katmanı denemeleri
│
├── nlp_veri_hatti.py         # Veri boru hattı - ilk sürüm
├── nlp_gercek_veri_hatti.py  # Veri boru hattı - iyileştirilmiş sürüm
├── nlp_tam_boru_hatti.py     # Tam veri boru hattı (Dataset/DataLoader, padding/truncation)
│
├── nlp_lstm_mimarisi.py      # BiLSTM model mimarisinin tanımı
├── nlp_model_mimarisi.py     # Model mimarisi - alternatif/iyileştirilmiş sürüm
├── nlp_endustriyel_model.py  # Model mimarisi - üretim seviyesi son sürüm
│
├── nlp_egitim_dongusu.py     # Eğitim döngüsü (training loop)
├── nlp_gercek_sinav.py       # Görülmemiş 25.000 satırlık test seti değerlendirmesi
├── nlp_buyuk_final.py        # Tüm bileşenleri bir araya getiren final betiği
├── nlp_huggingface_api.py    # Hugging Face entegrasyonu için alternatif servis betiği
│
└── .gitignore
```

> Not: `nlp_*` betiklerinin çoğu, aşağıdaki "Geliştirme Yolculuğu" bölümünde anlatılan AR-GE sürecinin farklı aşamalarını/denemelerini temsil eder; bu yüzden isimlerinde "temel", "güvenli", "profesyonel", "gerçek" gibi aşama belirten sıfatlar bulunur.

## 🛤️ Geliştirme Yolculuğu (Adım Adım İnşa)

Bu proje, hazır bir kütüphane fonksiyonunun tek satırda çağrıldığı bir yapı değil; veri setinden donanım entegrasyonuna kadar her adımın modüler olarak tasarlandığı bir AR-GE günlüğüdür:

| # | Aşama | İlgili Dosya |
|---|-------|--------------|
| 1 | Donanım ve altyapı testi — GPU (CUDA) kapasitesi ve VRAM erişiminin doğrulanması | `nlp_pytorch_test.py` |
| 2 | Veri boru hattı ve temizlik — ham IMDb yorumlarının Regex ile büyük/küçük harf, noktalama ve HTML etiketlerinden arındırılması | `nlp_metin_temizleme.py` |
| 3 | Özel sözlük inşası — frekans analiziyle donanım dostu 10.000 kelimelik bir embedding sözlüğü oluşturulması | `imdb_sozluk.json` |
| 4 | Matris paketlemesi — `Dataset`/`DataLoader` ile farklı uzunluktaki metinlerin 256'lık sabit tensörlere (padding/truncation) dönüştürülmesi | `nlp_tam_boru_hatti.py` |
| 5 | Model inşası ve eğitim — BiLSTM mimarisinin kurulması ve 25.000 yorum üzerinde eğitilerek ağırlıkların (`.pth`) kaydedilmesi | `nlp_egitim_dongusu.py` |
| 6 | Büyük yüzleşme — modelin hiç görmediği 25.000 satırlık test setiyle sınanması, overfitting kontrolü | `nlp_gercek_sinav.py` |
| 7 | Ürünleştirme — terminaldeki modelin FastAPI ile web servisine dönüştürülmesi ve asenkron bir arayüzle dış dünyaya açılması | `nlp_api.py`, `index.html` |

## 📊 Performans ve Kanıtlar

Model, eğitim sırasında hiç görmediği **25.000 satırlık test veri setinde** değerlendirilmiş ve aşağıdaki sonuçları elde etmiştir:

- **Gerçek Doğrulama Başarısı (Validation Accuracy):** **%83.70**
- Uzun, kinayeli ve karmaşık metinlerde (örn. profesyonel film eleştirileri) bağlamı koruyarak **%98'in üzerinde** eminlik oranlarına ulaşabilmektedir.

## 🛑 Bilinen Sınırlar ve Zayıf Yönler

Sistemin sınırlarını anlamak, gelişim sürecinin en kritik parçasıdır:

1. **Negation Scope (Olumsuzluk Kapsamı) Problemi:** LSTM mimarisinin bir "Dikkat" (Attention) mekanizmasına sahip olmaması nedeniyle, *"This movie is not good"* gibi kısa ve doğrudan zıtlık barındıran cümlelerde sistem hataya düşebilmektedir. "Good" kelimesinin güçlü pozitif matris ağırlığı, "not" kelimesinin negatif etkisini kısa bağlamlarda bastırabilmektedir.
2. **Karakter Filtreleme Kayıpları:** Regex tabanlı temizleme motoru (`[^a-z\s]`) Türkçe karakterleri ve noktalama işaretlerini tamamen sildiği için, kullanıcıdan gelen bazı girişlerde cümlenin gramer yapısı bozulabilmektedir.

## 💻 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.10+
- (Opsiyonel ama önerilir) CUDA destekli bir GPU

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/hakanEfeTuysuz/nlp-bilstm-sentiment-analysis.git
cd nlp-bilstm-sentiment-analysis

# 2. Gerekli kütüphaneleri kurun
pip install torch fastapi uvicorn pydantic

# 3. API sunucusunu başlatın
uvicorn nlp_api:app --reload
```

Ardından:
- Swagger UI üzerinden test etmek için tarayıcıda **http://127.0.0.1:8000/docs** adresine gidin.
- Ya da doğrudan `index.html` dosyasını tarayıcınızda açarak görsel arayüzü kullanın.

## 🛠️ Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Model / Eğitim | PyTorch (BiLSTM, Dropout) |
| Veri İşleme | Regex, özel tokenizer/sözlük |
| API | FastAPI, Uvicorn, Pydantic |
| Arayüz | HTML / CSS / JavaScript (Fetch API) |

## 🗺 Yol Haritası

- [ ] Negation-scope problemini çözmek için **Attention** mekanizması veya **BERT (Transformers)** tabanlı mimariye geçiş
- [ ] Türkçe karakter desteği içeren daha kapsamlı bir metin temizleme adımı
- [ ] Modelin Hugging Face üzerinde yayınlanması (`nlp_huggingface_api.py` bu yönde bir başlangıç noktasıdır)

## 🤝 Katkıda Bulunma

Katkılar, hata bildirimleri ve öneriler memnuniyetle karşılanır. Bir *issue* açabilir veya *pull request* gönderebilirsiniz.

## 📄 Lisans

Bu proje için henüz bir lisans belirlenmemiştir. Kullanmadan önce depo sahibiyle iletişime geçmeniz önerilir.
