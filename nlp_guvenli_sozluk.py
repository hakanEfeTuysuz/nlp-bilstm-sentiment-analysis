import torch

print("--- NLP Adım 2: Güvenli Sözlük ve Padding ---\n")

corpus = [
    "bu film gerçekten harika",
    "berbat bir film",
    "film fena değil",
    "harika bir senaryo"
]

# 1. Aşama: Sözlüğe Sistem Token'larını (Jokerleri) Ekleyerek Başlıyoruz
sozluk = {
    "<PAD>": 0,  # Kısa cümleleri uzatmak için dolgu
    "<UNK>": 1   # Bilinmeyen kelimeler için default sepet
}
index = 2  # Kelime ID'lerimiz artık 2'den başlıyor

for cumle in corpus:
    kelimeler = cumle.split()
    for kelime in kelimeler:
        if kelime not in sozluk:
            sozluk[kelime] = index
            index += 1

print("1. Güvenli Sözlüğümüz:")
print(sozluk)

# 2. Aşama: Bilinmeyen Kelime İçeren Test Cümlesi
test_cumlesi = "berbat bir yönetmen"
print(f"\nTest Cümlesi: '{test_cumlesi}'")

# .get() metodu kullanıyoruz: Eğer kelime sözlükte yoksa default olarak sozluk["<UNK>"] (yani 1) döner
sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in test_cumlesi.split()]
print(f"Çevrilen Dizi: {sayisal_dizi} -> (Dikkat et, 'yönetmen' kelimesi 1 ID'sini aldı)")

# 3. Aşama: Padding (Dolgu) İşlemi
# Yapay zekanın tüm cümleleri sabit bir uzunlukta (örneğin 5 kelime) görmesini istiyoruz
SABIT_UZUNLUK = 5

if len(sayisal_dizi) < SABIT_UZUNLUK:
    # Eksik kısım kadar <PAD> (0) ekliyoruz
    eksik_miktar = SABIT_UZUNLUK - len(sayisal_dizi)
    sayisal_dizi.extend([sozluk["<PAD>"]] * eksik_miktar)
elif len(sayisal_dizi) > SABIT_UZUNLUK:
    # Eğer cümle çok uzunsa fazlalığı kırpıyoruz
    sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]

# GPU'ya gitmeye hazır, kusursuz sabit uzunlukta PyTorch Tensor'u
vektor_tensor = torch.tensor(sayisal_dizi, dtype=torch.long)

print(f"\nPadding Sonrası Tensor (GPU'ya Hazır):")
print(vektor_tensor)