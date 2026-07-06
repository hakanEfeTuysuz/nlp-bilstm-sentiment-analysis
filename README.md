# 🎬 Endüstriyel NLP: BiLSTM Tabanlı Duygu Analizi (Sentiment Analysis)

Bu proje, IMDb film yorumlarını analiz ederek "Olumlu" veya "Olumsuz" olarak sınıflandıran, uçtan uca (Full-Stack) tasarlanmış bir Doğal Dil İşleme (NLP) sistemidir. Model sıfırdan eğitilmiş olup, FastAPI ile dış dünyaya açılmış ve modern bir web arayüzü ile desteklenmiştir.

## 🚀 Proje Mimarisi ve Özellikleri

* **Özel Veri Boru Hattı:** Ham metinler Regex ile temizlenmiş, kelime frekans analizi yapılarak donanım dostu 10.000 kelimelik özel bir sözlük (`imdb_sozluk.json`) inşa edilmiştir.
* **Donanım Optimizasyonu:** PyTorch `Dataset` ve `DataLoader` sınıfları yazılarak metinler 256 uzunluğunda tensor'lara dönüştürülmüş, CUDA (GPU) üzerinde darboğazsız bir eğitim sağlanmıştır.
* **Derin Öğrenme Modeli:** Cümle bağlamını hem ileri hem geri okuyabilen Çift Yönlü LSTM (BiLSTM) mimarisi kullanılmıştır. Ezberlemeyi (Overfitting) önlemek için %50 oranında `Dropout` katmanı sisteme entegre edilmiştir.
* **Production-Ready API:** Eğitilen model (`.pth`), FastAPI kullanılarak yüksek performanslı bir RESTful API'ye dönüştürülmüştür.
* **Kullanıcı Arayüzü:** API ile asenkron (Fetch API) haberleşen, karanlık temalı (Dark Mode) bir HTML/CSS/JS frontend yazılmıştır.

## 🛤️ Geliştirme Yolculuğu (Adım Adım İnşa)

Bu proje, hazır bir kütüphane fonksiyonunun tek satırda çağrıldığı bir yapı değil; veri setinden donanım entegrasyonuna kadar her adımın modüler olarak tasarlandığı bir AR-GE günlüğüdür:

1. **Donanım ve Altyapı Testi (`nlp_pytorch_test.py`):** Sistemin GPU (CUDA) kapasitesinin ve VRAM erişiminin doğrulanması.
2. **Veri Boru Hattı ve Temizlik (`nlp_metin_temizleme.py`):** Ham IMDb yorumlarının Regex ile işlenerek büyük/küçük harf, noktalama işaretleri ve HTML etiketlerinden arındırılması.
3. **Özel Sözlük İnşası (`imdb_sozluk.json`):** Temizlenen veriler üzerinde frekans analizi yapılarak, donanım dostu 10.000 kelimelik özel bir Embedding (Gömülme) sözlüğünün oluşturulması.
4. **Matris Paketlemesi (`nlp_tam_boru_hatti.py`):** PyTorch `Dataset` ve `DataLoader` kullanılarak farklı uzunluklardaki metinlerin ekran kartına 256'lık sabit tensor matrisleri halinde (Padding/Truncation) beslenmesi.
5. **Model İnşası ve Eğitim (`nlp_egitim_dongusu.py`):** BiLSTM mimarisinin kurulması ve modelin 25.000 yorum üzerinden eğitilerek öğrenilmiş ağırlıkların (`.pth`) diske kaydedilmesi.
6. **Büyük Yüzleşme (`nlp_gercek_sinav.py`):** Modelin daha önce hiç görmediği 25.000 satırlık test setiyle sınava sokulup, ezber (Overfitting) testinin yapılması.
7. **Ürünleştirme (`nlp_api.py` & Frontend):** Terminalde çalışan "beynin" FastAPI ile bir web servisine dönüştürülmesi ve asenkron çalışan bir HTML arayüzü ile dış dünyaya açılması.

## 📊 Performans ve Kanıtlar

Model, eğitim sırasında hiç görmediği **25.000 satırlık test veri setinde** değerlendirilmiş ve aşağıdaki sonuçları elde etmiştir:
* **Gerçek Doğrulama Başarısı (Validation Accuracy):** %83.70
* Uzun, kinayeli ve karmaşık metinlerde (örn. profesyonel film eleştirileri) bağlamı koruyarak %98'in üzerinde eminlik oranlarına ulaşabilmektedir.

## 🛑 Bilinen Sınırlar ve Zayıf Yönler (Known Limitations)

Sistemin sınırlarını anlamak, gelişim sürecinin en kritik parçasıdır. Modelin mimari kısıtlamaları şu şekildedir:

1. **Negation Scope (Olumsuzluk Kapsamı) Problemi:** LSTM mimarisinin "Dikkat" (Attention) mekanizmasına sahip olmaması nedeniyle, *"This movie is not good"* gibi kısa ve doğrudan zıtlık barındıran cümlelerde sistem çuvallayabilmektedir. "Good" kelimesinin devasa pozitif matris ağırlığı, "not" kelimesinin negatif etkisini kısa bağlamlarda ezip geçmektedir.
2. **Karakter Filtreleme Kayıpları:** Regex tabanlı temizleme motorumuz (`[^a-z\s]`), Türkçe karakterleri veya noktalama işaretlerini tamamen sildiği için, kullanıcıdan gelen hatalı girişlerde cümlenin gramer yapısı bozulabilmektedir. 

*Gelecek sürümlerde bu sorunları çözmek için BERT (Transformers) mimarisine geçilmesi planlanmaktadır.*

## 💻 Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri kurun: `pip install torch fastapi uvicorn pydantic`
2. API sunucusunu başlatın: `uvicorn nlp_api:app --reload`
3. Swagger UI üzerinden test etmek için tarayıcıda `http://127.0.0.1:8000/docs` adresine gidin.
4. Veya doğrudan `index.html` dosyasını tarayıcınızda açarak görsel arayüzü kullanın.
