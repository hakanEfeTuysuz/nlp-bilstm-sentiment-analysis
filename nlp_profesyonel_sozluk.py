import re
from collections import Counter
from datasets import load_dataset

print("--- NLP Adım 11: Profesyonel Sözlük (Vocabulary) İnşası ---\n")

print("1. Veri Seti Yükleniyor...")
imdb_verisi = load_dataset("stanfordnlp/imdb")

def metni_temizle(ham_metin):
    metin = ham_metin.lower()
    metin = re.sub(r'<[^>]+>', ' ', metin)
    metin = re.sub(r'[^a-z\s]', '', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()
    return metin

print("2. 25.000 Eğitim Yorumu Temizleniyor ve Kelimeler Sayılıyor...")
print("(Bu işlem işlemci gücüne bağlı olarak 10-15 saniye sürebilir...)\n")

# Tüm kelimelerin frekansını (kaç kere geçtiğini) tutacağımız özel sayaç nesnesi
kelime_sayaci = Counter()

# Sadece eğitim (train) setini kullanıyoruz! 
# KURAL: Modelin test setindeki kelimeleri önceden görmemesi (kopya çekmemesi) gerekir.
for yorum in imdb_verisi['train']['text']:
    temiz_metin = metni_temizle(yorum)
    kelimeler = temiz_metin.split()
    # Kelimeleri sayaca ekle
    kelime_sayaci.update(kelimeler)

toplam_essiz_kelime = len(kelime_sayaci)
print(f"-> Analiz Tamamlandı! Toplam {toplam_essiz_kelime} farklı (eşsiz) kelime bulundu.\n")

print("3. Maksimum Kapasite Filtresi Uygulanıyor (Top 10.000 Kelime)")
MAKS_KELIME = 10000

# En çok geçen 10.000 kelimeyi çekiyoruz
en_sik_kelimeler = kelime_sayaci.most_common(MAKS_KELIME)

# Sözlüğümüzü standart jokerlerimizle (<PAD> ve <UNK>) başlatıyoruz
sozluk = {"<PAD>": 0, "<UNK>": 1}
index = 2

for kelime, frekans in en_sik_kelimeler:
    sozluk[kelime] = index
    index += 1

print(f"-> Profesyonel Sözlük Hazır! Nihai Boyut: {len(sozluk)}\n")

print("En Çok Kullanılan İlk 5 Kelime (İngilizcenin Yapıtaşları):")
for kelime, frekans in en_sik_kelimeler[:5]:
    print(f"  '{kelime}': {frekans} kez")

print("\nEn Az Kullanılan (10.000. Sınırda Kalan) 5 Kelime:")
for kelime, frekans in en_sik_kelimeler[-5:]:
    print(f"  '{kelime}': {frekans} kez")