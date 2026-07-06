import torch
import torch.nn as nn
import json
import re
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

print("--- NLP Adım 14: Büyük Yüzleşme (Görünmeyen 25.000 Test Verisi) ---\n")

# 1. DONANIM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Sistem {device.type.upper()} üzerinden çalıştırılıyor...")

# 2. SÖZLÜĞÜ VE VERİYİ YÜKLEME
print("-> Sözlük ve Test Verisi Yükleniyor...")
with open('imdb_sozluk.json', 'r', encoding='utf-8') as f:
    sozluk = json.load(f)

imdb_verisi = load_dataset("stanfordnlp/imdb")

def metni_temizle(ham_metin):
    metin = ham_metin.lower()
    metin = re.sub(r'<[^>]+>', ' ', metin)
    metin = re.sub(r'[^a-z\s]', '', metin)
    return re.sub(r'\s+', ' ', metin).strip()

# 3. KUSURSUZ KARGO BANDI (Bu kez test verisi için)
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

# DİKKAT: Burada 'train' değil 'test' verisini kullanıyoruz!
test_kargosu = DataLoader(IMDbDataset(imdb_verisi['test']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=veri_paketleyici)

# 4. MİMARİYİ TANIMLAMA VE BEYNİ YÜKLEME
class LSTMDuyguModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu, gizli_katman):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu, padding_idx=0)
        self.lstm = nn.LSTM(input_size=vektor_boyutu, hidden_size=gizli_katman, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(gizli_katman * 2, 2)

    def forward(self, x):
        gomumler = self.embedding(x)
        lstm_cikis, (hidden, cell) = self.lstm(gomumler)
        ileri_hafiza = hidden[-2]
        geri_hafiza = hidden[-1]
        son_hafiza = torch.cat((ileri_hafiza, geri_hafiza), dim=1)
        duzenlenmis_hafiza = self.dropout(son_hafiza)
        return self.fc(duzenlenmis_hafiza)

model = LSTMDuyguModeli(sozluk_boyutu=len(sozluk), vektor_boyutu=64, gizli_katman=64).to(device)
model.load_state_dict(torch.load("imdb_lstm_modeli.pth"))

# 5. GERÇEK SINAV (EVALUATION)
print("\n--- Model 25.000 Yeni Soruyla Sınava Giriyor ---")
model.eval() # Modeli sınav moduna al (Ezberleme ve öğrenme kapalı)

dogru_tahmin = 0
toplam_veri = 0

with torch.no_grad(): # Türev hesaplamayı kapat (GPU çok hızlı çalışacak)
    for X_batch, Y_batch in test_kargosu:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        tahminler = model(X_batch)
        secilen_siniflar = torch.argmax(tahminler, dim=1)
        
        dogru_tahmin += (secilen_siniflar == Y_batch).sum().item()
        toplam_veri += Y_batch.size(0)

gercek_basari_orani = (dogru_tahmin / toplam_veri) * 100

print("-" * 50)
print(f"Toplam Test Edilen Yorum: {toplam_veri}")
print(f"Doğru Bilinen Yorum: {dogru_tahmin}")
print(f"MODELİN GERÇEK BAŞARISI (DOĞRULAMA): %{gercek_basari_orani:.2f}")
print("-" * 50)

if gercek_basari_orani >= 80:
    print("\nSonuç: Harika! Model genel kuralları gerçekten öğrenmiş, ezber minimumda.")
else:
    print("\nSonuç: Model eğitim verisini fazla ezberlemiş (Overfitting). Daha fazla Dropout veya erken durdurma (Early Stopping) gerekebilir.")