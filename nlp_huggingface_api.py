from datasets import load_dataset

print("--- NLP Adım 9: Hugging Face API ile Gerçek Veriye Geçiş ---\n")

print("1. IMDb Veri Seti Sunuculardan Çekiliyor...")
print("(Bu işlem internet hızına bağlı olarak ilk seferde 1-2 dakika sürebilir, sonrasında önbelleğe alınır.)\n")

# Bütün o 50.000 satırlık devasa veri setini tek satırda indiriyoruz
imdb_verisi = load_dataset("stanfordnlp/imdb")

print("2. Veri Seti Başarıyla Yüklendi ve Parçalandı!")
# API bizim için veriyi eğitim (train) ve test (test) olarak ayırdı bile
print(imdb_verisi)

print("\n--------------------------------------------------")
print("3. Laboratuvardan Çıkış: Gerçek Bir Veri İncelemesi")
print("--------------------------------------------------\n")

# Eğitim setindeki ilk yoruma (0. indeks) göz atalım
ornek_yorum = imdb_verisi['train'][0]
ornek_metin = ornek_yorum['text']
ornek_etiket = ornek_yorum['label']

durum = "Olumlu" if ornek_etiket == 1 else "Olumsuz"

print(f"Etiket: {ornek_etiket} ({durum})\n")
print(f"Yorumun Tamamı (Karakter Sayısı: {len(ornek_metin)}):\n")
print(ornek_metin)

print("\n4. Bu Metni Eski Yöntemle Böldüğümüzde Çıkan Kelime Sayısı:")
print(len(ornek_metin.split()))