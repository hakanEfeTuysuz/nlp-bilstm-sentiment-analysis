import torch
import torch.nn as nn

print("--- NLP Adım 3: Embedding (Kelime Vektörleri) Katmanı ---\n")

# Bir önceki adımdaki sözlüğümüzün büyüklüğü (0'dan 10'a kadar toplam 11 ID)
sozluk_boyutu = 11

# Her kelimeyi kaç boyutlu bir uzayda temsil edeceğiz?
# Gerçek projelerde bu genelde 128, 256 veya 512 olur. Biz görmek için 3 yapıyoruz.
vektor_boyutu = 3

# 1. PyTorch Embedding Katmanını Tanımlıyoruz
# Bu aslında arka planda devasa bir (11x3) ağırlık (weight) matrisidir!
embedding_katmani = nn.Embedding(num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu)

# 2. Bir önceki adımdan gelen padlenmiş tensorumuz:
# "berbat(6) bir(7) yönetmen(1) <PAD>(0) <PAD>(0)"
girdi_tensoru = torch.tensor([6, 7, 1, 0, 0], dtype=torch.long)

# 3. İleri Besleme (Forward Pass): ID'leri vektörlere dönüştürüyoruz
kelime_vektorleri = embedding_katmani(girdi_tensoru)

print("1. Girdi Tensoru (Kelime ID'leri):")
print(girdi_tensoru)

print("\n2. Embedding Sonrası Tensor (Kelime Vektörleri):")
print(kelime_vektorleri)

print(f"\n3. Yeni Matris Boyutu: {kelime_vektorleri.shape}")
print("Matrisin Anlamı: (Cümledeki kelime sayısı, Her kelimenin vektör boyutu)")