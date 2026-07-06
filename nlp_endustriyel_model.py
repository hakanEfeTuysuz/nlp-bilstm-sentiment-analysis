import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import Dataset, DataLoader

print("--- NLP Adım 8: Uçtan Uca Endüstriyel Model Eğitimi ---\n")

# 1. PARAMETRELER VE DOSYA
DOSYA_ADI = "film_yorumlari_veri_seti.csv"
SABIT_UZUNLUK = 6
BATCH_SIZE = 16
EPOCH_SAYISI = 20  # Aşırı öğrenmeyi önlemek için düşük tutuyoruz

# 2. SÖZLÜK OLUŞTURMA
sozluk = {"<PAD>": 0, "<UNK>": 1}
index = 2
with open(DOSYA_ADI, mode='r', encoding='utf-8') as f:
    okuyucu = csv.reader(f)
    next(okuyucu)
    for satir in okuyucu:
        for kelime in satir[0].split():
            if kelime not in sozluk:
                sozluk[kelime] = index
                index += 1

# 3. VERİ BORU HATTI (Dataset & DataLoader)
class FilmYorumlariDataset(Dataset):
    def __init__(self, csv_dosyasi):
        self.veriler = []
        with open(csv_dosyasi, mode='r', encoding='utf-8') as f:
            okuyucu = csv.reader(f)
            next(okuyucu)
            for satir in okuyucu:
                self.veriler.append((satir[0], int(satir[1])))
                
    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        return self.veriler[idx]

def veri_paketleyici(gelen_batch):
    islenmis_metinler = []
    etiketler = []
    for metin, etiket in gelen_batch:
        sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in metin.split()]
        if len(sayisal_dizi) < SABIT_UZUNLUK:
            sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
        else:
            sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
        islenmis_metinler.append(sayisal_dizi)
        etiketler.append(etiket)
    return torch.tensor(islenmis_metinler, dtype=torch.long), torch.tensor(etiketler, dtype=torch.long)

veri_seti = FilmYorumlariDataset(DOSYA_ADI)
veri_kargosu = DataLoader(dataset=veri_seti, batch_size=BATCH_SIZE, shuffle=True, collate_fn=veri_paketleyici)

# 4. MODEL MİMARİSİ
class DuyguAnaliziModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu, gizli_katman_boyutu, sinif_sayisi):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu, padding_idx=0)
        self.fc1 = nn.Linear(vektor_boyutu, gizli_katman_boyutu)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(gizli_katman_boyutu, sinif_sayisi)

    def forward(self, x):
        gomumler = self.embedding(x)
        cumle_vektoru = gomumler.mean(dim=1)
        return self.fc2(self.relu(self.fc1(cumle_vektoru)))

# Model, Kayıp Fonksiyonu ve Optimizatör (Kapasiteyi dengeli tuttuk: gizli_katman=8)
model = DuyguAnaliziModeli(sozluk_boyutu=len(sozluk), vektor_boyutu=8, gizli_katman_boyutu=8, sinif_sayisi=2)
kayip_fonksiyonu = nn.CrossEntropyLoss()
# Optimizatöre L2 cezası (weight_decay) ekledik ki bir kelimeye aşırı odaklanmasın
optimizator = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# 5. EĞİTİM DÖNGÜSÜ
print("Eğitim Başlıyor...\n")
for epoch in range(EPOCH_SAYISI):
    toplam_kayip = 0
    # Artık tüm veriyi tek seferde değil, DataLoader'dan paket paket alıyoruz!
    for X_batch, Y_batch in veri_kargosu:
        tahminler = model(X_batch)
        kayip = kayip_fonksiyonu(tahminler, Y_batch)
        
        optimizator.zero_grad()
        kayip.backward()
        optimizator.step()
        
        toplam_kayip += kayip.item()
        
    ortalama_kayip = toplam_kayip / len(veri_kargosu)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCH_SAYISI} | Ortalama Kayıp (Loss): {ortalama_kayip:.4f}")

print("\n--- Sistem Mimarisi Başarıyla Eğitildi! ---")