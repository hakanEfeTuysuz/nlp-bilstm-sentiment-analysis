# 🎬 Endüstriyel NLP: BiLSTM Tabanlı Duygu Analizi (Sentiment Analysis)

Bu proje, IMDb film yorumlarını analiz ederek "Olumlu" veya "Olumsuz" olarak sınıflandıran, uçtan uca (Full-Stack) tasarlanmış bir Doğal Dil İşleme (NLP) sistemidir. Model sıfırdan eğitilmiş olup, FastAPI ile dış dünyaya açılmış ve modern bir web arayüzü ile desteklenmiştir.

## 🚀 Proje Mimarisi ve Özellikleri

* **Özel Veri Boru Hattı:** Ham metinler Regex ile temizlenmiş, kelime frekans analizi yapılarak donanım dostu 10.000 kelimelik özel bir sözlük (`imdb_sozluk.json`) inşa edilmiştir.
* **Donanım Optimizasyonu:** PyTorch `Dataset` ve `DataLoader` sınıfları yazılarak metinler 256 uzunluğunda tensor'lara dönüştürülmüş, CUDA (GPU) üzerinde darboğazsız bir eğitim sağlanmıştır.
* **Derin Öğrenme Modeli:** Cümle bağlamını hem ileri hem geri okuyabilen Çift Yönlü LSTM (BiLSTM) mimarisi kullanılmıştır. Ezberlemeyi (Overfitting) önlemek için %50 oranında `Dropout` katmanı sisteme entegre edilmiştir.
* **Production-Ready API:** Eğitilen model (`.pth`), FastAPI kullanılarak yüksek performanslı bir RESTful API'ye dönüştürülmüştür.
* **Kullanıcı Arayüzü:** API ile asenkron (Fetch API) haberleşen, karanlık temalı (Dark Mode) bir HTML/CSS/JS frontend yazılmıştır.

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