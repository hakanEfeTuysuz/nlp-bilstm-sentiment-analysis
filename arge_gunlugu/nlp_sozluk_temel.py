import torch

print("--- NLP Adım 1: Sözlük (Vocabulary) İnşası ---\n")

# Gerçek dünyada bu veriler binlerce satırlık IMDb veya Twitter yorumlarından gelir.
# Biz mimariyi anlamak için mikro bir veri seti (corpus) kullanıyoruz.
corpus = [
    "bu film gerçekten harika",
    "berbat bir film",
    "film fena değil",
    "harika bir senaryo"
]

# 1. Aşama: Tüm kelimeleri çıkarıp eşsiz bir sözlük (Vocabulary) oluşturma
sozluk = {}
index = 0

for cumle in corpus:
    # Cümleyi boşluklardan kelimelere bölüyoruz (Basit Tokenization)
    kelimeler = cumle.split()
    for kelime in kelimeler:
        # Eğer kelime sözlükte yoksa, ona yeni bir ID atıyoruz
        if kelime not in sozluk:
            sozluk[kelime] = index
            index += 1

print("1. İnşa Edilen Eşsiz Sözlük:")
for kelime, id_no in sozluk.items():
    print(f"  '{kelime}' -> {id_no}")

# 2. Aşama: Yeni bir cümleyi matematiksel vektöre çevirme
ornek_cumle = "harika bir film"
# Cümledeki her kelimenin sözlükteki ID'sini bulup bir listeye koyuyoruz
sayisal_dizi = [sozluk[kelime] for kelime in ornek_cumle.split()]

# 3. Aşama: Bu diziyi PyTorch'un anlayacağı Tensor yapısına çevirip GPU'ya hazır hale getirme
# dtype=torch.long kullanıyoruz çünkü kelime ID'leri tam sayıdır (0, 1, 2...)
vektor_tensor = torch.tensor(sayisal_dizi, dtype=torch.long)

print(f"\n2. Veri Dönüşümü:")
print(f"  Orijinal Cümle : '{ornek_cumle}'")
print(f"  PyTorch Tensor : {vektor_tensor}")