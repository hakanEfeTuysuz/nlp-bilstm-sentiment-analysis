import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import json

print("--- NLP Adım 13: LSTM Zaman Serisi Mimarisi ile Kesin Çözüm ---\n")

# 1. DONANIM (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> İşlem Birimi: {device.type.upper()}\n")

# 2. VERİ VE SÖZLÜK (Boru Hattımız Aynı Şekilde Çalışıyor)
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
BATCH_SIZE = 64

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


# 4. YENİ MİMARİ: ÇİFT YÖNLÜ (BIDIRECTIONAL) LSTM
class LSTMDuyguModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu, gizli_katman):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu, padding_idx=0)
        
        # bidirectional=True ile model artık metni hem baştan hem sondan okuyacak
        self.lstm = nn.LSTM(input_size=vektor_boyutu, hidden_size=gizli_katman, batch_first=True, bidirectional=True)
        
        # Ezberlemeyi önlemek için %50 unutma filtresi
        self.dropout = nn.Dropout(0.5)
        
        # BiLSTM kullandığımız için ileri ve geri hafızalar birleşir (64 * 2 = 128 boyut)
        self.fc = nn.Linear(gizli_katman * 2, 2)

    def forward(self, x):
        gomumler = self.embedding(x)
        lstm_cikis, (hidden, cell) = self.lstm(gomumler)
        
        # Çift yönlü olduğu için son iki hafızayı (ileri ve geri) birleştiriyoruz
        ileri_hafiza = hidden[-2]
        geri_hafiza = hidden[-1]
        son_hafiza = torch.cat((ileri_hafiza, geri_hafiza), dim=1)
        
        # Dropout uygulayıp karar katmanına gönderiyoruz
        duzenlenmis_hafiza = self.dropout(son_hafiza)
        return self.fc(duzenlenmis_hafiza)

model = LSTMDuyguModeli(sozluk_boyutu=len(sozluk), vektor_boyutu=64, gizli_katman=64).to(device)

# İŞTE EKSİK OLAN SATIR: Kayıp fonksiyonumuzu tekrar tanımlıyoruz
kayip_fonksiyonu = nn.CrossEntropyLoss()

# Optimizatörün hızı düşürüldü (Daha hassas adımlar atacak)
optimizator = optim.Adam(model.parameters(), lr=0.001)

# 5. EĞİTİM DÖNGÜSÜ
EPOCH_SAYISI = 15
print("\n--- LSTM Modeli Eğitiliyor (Sıralı Okuma Aktif) ---")

for epoch in range(EPOCH_SAYISI):
    toplam_kayip = 0
    dogru_tahmin = 0
    toplam_veri = 0
    
    for X_batch, Y_batch in egitim_kargosu:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        tahminler = model(X_batch)
        kayip = kayip_fonksiyonu(tahminler, Y_batch)
        
        optimizator.zero_grad()
        kayip.backward()
        optimizator.step()
        
        toplam_kayip += kayip.item()
        
        secilen_siniflar = torch.argmax(tahminler, dim=1)
        dogru_tahmin += (secilen_siniflar == Y_batch).sum().item()
        toplam_veri += Y_batch.size(0)
        
    ortalama_kayip = toplam_kayip / len(egitim_kargosu)
    basari_orani = (dogru_tahmin / toplam_veri) * 100
    print(f"Epoch {epoch+1}/{EPOCH_SAYISI} | Kayıp: {ortalama_kayip:.4f} | LSTM Başarısı: %{basari_orani:.2f}")

print("\n--- Eğitim Tamamlandı! ---")
print("\n--- Model ve Sözlük Diske Kaydediliyor ---")

# 1. Modelin Öğrendiği Ağırlıkları (Matrisleri) Kaydetme
# .pth veya .pt uzantısı PyTorch'un standart kayıt formatıdır.
torch.save(model.state_dict(), "imdb_lstm_modeli.pth")

# 2. Modelin Dilini (Sözlüğünü) Kaydetme
with open('imdb_sozluk.json', 'w', encoding='utf-8') as f:
    json.dump(sozluk, f)

print("Başarılı: 'imdb_lstm_modeli.pth' ve 'imdb_sozluk.json' dosyaları oluşturuldu!")

# 6. ZORLU CANLI TEST
print("\n--- LSTM Bağlam Testi ---")
test_cumleleri = [
    "This movie was absolutely wonderful, I loved every second of it!",
    "Terrible acting, awful script, complete waste of time.",
    "It was okay, not the best but watchable.", # O meşhur zorlu cümle!
    "I fell asleep halfway through, so boring."
]

model.eval()
with torch.no_grad():
    for metin in test_cumleleri:
        temiz_metin = metni_temizle(metin)
        sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in temiz_metin.split()]
        if len(sayisal_dizi) < SABIT_UZUNLUK:
            sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
        else:
            sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
            
        x_tensor = torch.tensor([sayisal_dizi], dtype=torch.long).to(device)
        cikis = model(x_tensor)
        
        tahmin = torch.argmax(cikis, dim=1).item()
        olasilik = torch.softmax(cikis, dim=1)[0][tahmin].item() * 100
        durum = "Olumlu" if tahmin == 1 else "Olumsuz"
        
        print(f"\nMetin: '{metin}'")
        print(f"Tahmin: {durum} (Eminlik Oranı: %{olasilik:.2f})")