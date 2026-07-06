import torch
import torch.nn as nn
import torch.optim as optim

print("--- NLP Adım 5.1: Düzeltilmiş Mimarisi ve Eğitim ---")

# 1. VERİ SETİ (Biraz daha zenginleştirildi ki ezberleme bozulsun)
corpus = [
    ("bu film gerçekten harika", 1),
    ("berbat bir film", 0),
    ("film fena değil", 1),
    ("harika bir senaryo", 1),
    ("berbat ve kötü senaryo", 0),
    ("bu gerçekten kötü", 0),
    ("gerçekten çok güzel", 1),  # Yeni
    ("tamamen berbat", 0),       # Yeni
    ("kötü film", 0),            # Yeni
    ("harika", 1)                # Yeni
]

sozluk = {"<PAD>": 0, "<UNK>": 1}
index = 2
for cumle, etiket in corpus:
    for kelime in cumle.split():
        if kelime not in sozluk:
            sozluk[kelime] = index
            index += 1

SABIT_UZUNLUK = 5

def veriyi_hazirla(metin):
    dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in metin.split()]
    if len(dizi) < SABIT_UZUNLUK:
        dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(dizi)))
    else:
        dizi = dizi[:SABIT_UZUNLUK]
    return dizi

X_veri = torch.tensor([veriyi_hazirla(cumle) for cumle, etiket in corpus], dtype=torch.long)
Y_etiket = torch.tensor([etiket for cumle, etiket in corpus], dtype=torch.long)

# 2. MODEL MİMARİSİ (Hata Giderildi)
class DuyguAnaliziModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu, gizli_katman_boyutu, sinif_sayisi):
        super().__init__()
        
        # MİMARİ DÜZELTME: padding_idx=0 ile PAD vektörünü [0,0,0] olarak donduruyoruz!
        self.embedding = nn.Embedding(num_embeddings=sozluk_boyutu, 
                                      embedding_dim=vektor_boyutu, 
                                      padding_idx=0)
        
        self.fc1 = nn.Linear(vektor_boyutu, gizli_katman_boyutu)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(gizli_katman_boyutu, sinif_sayisi)

    def forward(self, x):
        gomumler = self.embedding(x)
        # Artık [0,0,0] olan PAD'ler toplama etki etmeyecek (Sadece 5'e bölündüğü için yoğunluğu azaltacak)
        cumle_vektoru = gomumler.mean(dim=1)
        return self.fc2(self.relu(self.fc1(cumle_vektoru)))

model = DuyguAnaliziModeli(sozluk_boyutu=len(sozluk), vektor_boyutu=8, gizli_katman_boyutu=8, sinif_sayisi=2)

kayip_fonksiyonu = nn.CrossEntropyLoss()
optimizator = optim.Adam(model.parameters(), lr=0.01)

# 3. EĞİTİM
epoch_sayisi = 50
print("\nEğitim Başlıyor...")
for epoch in range(epoch_sayisi):
    tahminler = model(X_veri)
    kayip = kayip_fonksiyonu(tahminler, Y_etiket)
    optimizator.zero_grad()
    kayip.backward()
    optimizator.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epoch_sayisi} | Kayıp (Loss): {kayip.item():.4f}")

# 4. CANLI TEST
test_cumleleri = ["harika film", "gerçekten berbat", "fena senaryo"]

print("\nCanlı Test Sonuçları:")
with torch.no_grad():
    for test in test_cumleleri:
        x_test = torch.tensor([veriyi_hazirla(test)], dtype=torch.long)
        cikis = model(x_test)
        tahmin_edilen_sinif = torch.argmax(cikis, dim=1).item()
        durum = "Olumlu" if tahmin_edilen_sinif == 1 else "Olumsuz"
        print(f"Metin: '{test}' -> Tahmin: {durum}")