import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

print("--- NLP BÜYÜK FİNAL: IMDb Duygu Analizi Modeli ---\n")

# 1. DONANIM HIZLANDIRMA (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Sistem {device.type.upper()} üzerinden çalıştırılıyor...\n")

# 2. VERİ VE SÖZLÜK HAZIRLIĞI
print("-> Veriler yükleniyor ve Sözlük inşa ediliyor...")
imdb_verisi = load_dataset("stanfordnlp/imdb")

def metni_temizle(ham_metin):
    metin = ham_metin.lower()
    metin = re.sub(r'<[^>]+>', ' ', metin)
    metin = re.sub(r'[^a-z\s]', '', metin)
    return re.sub(r'\s+', ' ', metin).strip()

kelime_sayaci = Counter()
for yorum in imdb_verisi['train']['text']:
    kelime_sayaci.update(metni_temizle(yorum).split())

sozluk = {"<PAD>": 0, "<UNK>": 1}
index = 2
for kelime, _ in kelime_sayaci.most_common(10000):
    sozluk[kelime] = index
    index += 1

# 3. VERİ BORU HATTI
SABIT_UZUNLUK = 256
BATCH_SIZE = 64 # GPU'ya aynı anda 64 yorum yollayacağız

class IMDbDataset(Dataset):
    def __init__(self, hf_dataset):
        self.veriler = hf_dataset
    def __len__(self):
        return len(self.veriler)
    def __getitem__(self, idx):
        return metni_temizle(self.veriler[idx]['text']), self.veriler[idx]['label']

def veri_paketleyici(gelen_batch):
    islenmis_metinler, etiketler = [], []
    for metin, etiket in gelen_batch:
        sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in metin.split()]
        if len(sayisal_dizi) < SABIT_UZUNLUK:
            sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
        else:
            sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
        islenmis_metinler.append(sayisal_dizi)
        etiketler.append(etiket)
    return torch.tensor(islenmis_metinler, dtype=torch.long), torch.tensor(etiketler, dtype=torch.long)

egitim_kargosu = DataLoader(IMDbDataset(imdb_verisi['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=veri_paketleyici)

# 4. SİNİR AĞI MİMARİSİ
class GercekDuyguModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu, gizli_katman):
        super().__init__()
        # Vektör boyutunu 3'ten 64'e çıkardık, kelimeler artık 64 boyutlu uzayda!
        self.embedding = nn.Embedding(num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu, padding_idx=0)
        self.fc1 = nn.Linear(vektor_boyutu, gizli_katman)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(gizli_katman, 2)

    def forward(self, x):
        gomumler = self.embedding(x)
        # 256 kelimelik vektörlerin ortalamasını alıp tek bir 'cümle vektörü' yapıyoruz
        cumle_vektoru = gomumler.mean(dim=1)
        return self.fc2(self.relu(self.fc1(cumle_vektoru)))

# Modeli GPU'ya gönderiyoruz
model = GercekDuyguModeli(sozluk_boyutu=len(sozluk), vektor_boyutu=64, gizli_katman=32).to(device)
kayip_fonksiyonu = nn.CrossEntropyLoss()
optimizator = optim.Adam(model.parameters(), lr=0.005)

# 5. EĞİTİM DÖNGÜSÜ
EPOCH_SAYISI = 5
print("\n--- Model Eğitimi Başlıyor (Matris Çarpımları Ateşlendi) ---")

for epoch in range(EPOCH_SAYISI):
    toplam_kayip = 0
    dogru_tahmin = 0
    toplam_veri = 0
    
    for X_batch, Y_batch in egitim_kargosu:
        # Verileri GPU'ya gönderiyoruz
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        tahminler = model(X_batch)
        kayip = kayip_fonksiyonu(tahminler, Y_batch)
        
        optimizator.zero_grad()
        kayip.backward()
        optimizator.step()
        
        toplam_kayip += kayip.item()
        
        # Başarı Oranını (Accuracy) Hesaplama
        secilen_siniflar = torch.argmax(tahminler, dim=1)
        dogru_tahmin += (secilen_siniflar == Y_batch).sum().item()
        toplam_veri += Y_batch.size(0)
        
    ortalama_kayip = toplam_kayip / len(egitim_kargosu)
    basari_orani = (dogru_tahmin / toplam_veri) * 100
    print(f"Epoch {epoch+1}/{EPOCH_SAYISI} | Kayıp: {ortalama_kayip:.4f} | Eğitim Başarısı: %{basari_orani:.2f}")

print("\n--- Eğitim Tamamlandı! ---")

# 6. CANLI TEST (Senin Cümlelerin)
print("\n--- Model Test Ediliyor ---")
test_cumleleri = [
    "This movie was absolutely wonderful, I loved every second of it!",
    "Terrible acting, awful script, complete waste of time.",
    "It was okay, not the best but watchable.",
    "I fell asleep halfway through, so boring."
]

model.eval() # Modeli test moduna alıyoruz (Dropout vb. varsa kapanır)
with torch.no_grad():
    for metin in test_cumleleri:
        temiz_metin = metni_temizle(metin)
        sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in temiz_metin.split()]
        if len(sayisal_dizi) < SABIT_UZUNLUK:
            sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
        else:
            sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
            
        # 1 adet cümleyi [1, 256] formatında GPU'ya yolluyoruz
        x_tensor = torch.tensor([sayisal_dizi], dtype=torch.long).to(device)
        cikis = model(x_tensor)
        
        tahmin = torch.argmax(cikis, dim=1).item()
        olasilik = torch.softmax(cikis, dim=1)[0][tahmin].item() * 100
        durum = "Olumlu" if tahmin == 1 else "Olumsuz"
        
        print(f"\nMetin: '{metin}'")
        print(f"Tahmin: {durum} (Eminlik Oranı: %{olasilik:.2f})")