import torch
import torch.nn as nn

print("--- NLP Adım 4: PyTorch Model Mimarisi (OOP) ---\n")

# 1. Model Sınıfının Tanımlanması
class DuyguAnaliziModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu, gizli_katman_boyutu, sinif_sayisi):
        super().__init__() # Üst sınıfın (nn.Module) gücünü miras alıyoruz

        # 1. Katman: Kelimeleri vektörlere (sayılara) çeviren donanım
        self.embedding = nn.Embedding(num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu)

        # 2. Katman: CNN'lerden çok iyi bildiğin Karar Mekanizması (Tam Bağlı / Linear Katman)
        self.fc1 = nn.Linear(vektor_boyutu, gizli_katman_boyutu)
        self.relu = nn.ReLU() # Doğrusallığı kıran aktivasyon fonksiyonumuz
        
        # 3. Çıktı Katmanı: Sonuçları 2 sınıfa (Olumlu=1, Olumsuz=0) indirgiyoruz
        self.fc2 = nn.Linear(gizli_katman_boyutu, sinif_sayisi)

    # İleri Besleme Matematiği
    def forward(self, x):
        # x'in boyutu: (1, 5) -> 1 cümle, 5 kelime (ID'ler)

        # 1. Adım: ID'leri Vektörlere Çevir
        gomumler = self.embedding(x) 
        # gomumler boyutu: (1, 5, 3) -> 1 cümle, 5 kelime, her kelime 3 boyutlu vektör

        # 2. Adım: Sıkıştırma (Cümle Vektörü Oluşturma)
        # Ağ, 5 ayrı kelime vektörü yerine cümlenin genel anlamına bakmalı.
        # Bu yüzden 5 kelime matrisinin ortalamasını (mean) alıp tek bir satıra indirgiyoruz.
        cumle_vektoru = gomumler.mean(dim=1) 
        # cumle_vektoru boyutu: (1, 3) -> 1 cümle, 3 boyutlu tek bir anlam vektörü

        # 3. Adım: Klasik Sinir Ağı Karar Süreci
        gizli_cikis = self.fc1(cumle_vektoru)
        aktif_cikis = self.relu(gizli_cikis)
        sonuc = self.fc2(aktif_cikis) # Boyut: (1, 2) -> 2 sınıf için puanlar

        return sonuc

# 2. Mimariyi Test Edelim
# sozluk_boyutu=11 (Önceki adımdan), vektor_boyutu=3, gizli_katman=8 nöron, sinif=2 (Olumlu/Olumsuz)
model = DuyguAnaliziModeli(sozluk_boyutu=11, vektor_boyutu=3, gizli_katman_boyutu=8, sinif_sayisi=2)

# Girdi: "berbat(6) bir(7) yönetmen(1) <PAD>(0) <PAD>(0)"
girdi_tensoru = torch.tensor([[6, 7, 1, 0, 0]], dtype=torch.long)

# Modeli çalıştırıyoruz (Arka planda forward metodu tetiklenir)
tahmin_puanlari = model(girdi_tensoru)

print("Girdi Cümlesi: 'berbat bir yönetmen'")
print("\nModelin Karar Çıktısı (Ham Puanlar / Logits):")
print(tahmin_puanlari)