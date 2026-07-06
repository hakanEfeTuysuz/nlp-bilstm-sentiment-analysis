from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # YENİ EKLENDİ
import torch
import torch.nn as nn
import json
import re

print("--- Sistem Başlatılıyor: NLP API Motoru Isınıyor ---")

app = FastAPI(
    title="IMDb NLP Yapay Zeka API",
    description="BiLSTM mimarisi ile eğitilmiş, metinlerin duygusunu analiz eden profesyonel web servisi.",
    version="1.0.0"
)

# YENİ EKLENDİ: Tarayıcıların API'ye erişmesine izin veren güvenlik kapısı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Her yerden gelen isteklere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2. SÖZLÜK VE DONANIM HAZIRLIĞI
with open('imdb_sozluk.json', 'r', encoding='utf-8') as f:
    sozluk = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. MİMARİYİ TANIMLAMA (Aşçının Yetenekleri)
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
        son_hafiza = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(son_hafiza))

# 4. BEYNİ YÜKLEME VE DONDURMA
model = LSTMDuyguModeli(sozluk_boyutu=len(sozluk), vektor_boyutu=64, gizli_katman=64).to(device)
# map_location ile modelin RAM'de mi yoksa VRAM'de mi çalışacağını garantiye alıyoruz
model.load_state_dict(torch.load("imdb_lstm_modeli.pth", map_location=device, weights_only=True))
model.eval() # Eğitimi kapat, sadece hizmet ver!

print("-> Model başarıyla belleğe yüklendi. API istek bekliyor...")

# 5. GİRİŞ ŞABLONU (Kullanıcıdan ne bekliyoruz?)
class YorumIstegi(BaseModel):
    metin: str

# 6. ENDPOINT (Kullanıcının İstek Atacağı URL Kapısı)
@app.post("/analiz-et")
def duyguyu_analiz_et(istek: YorumIstegi):
    # a. Metni Temizle
    temiz_metin = re.sub(r'[^a-z\s]', '', re.sub(r'<[^>]+>', ' ', istek.metin.lower()))
    temiz_metin = re.sub(r'\s+', ' ', temiz_metin).strip()
    
    # b. Sayılara Çevir ve 256 Uzunluğa Sabitle
    sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in temiz_metin.split()]
    SABIT_UZUNLUK = 256
    
    if len(sayisal_dizi) < SABIT_UZUNLUK:
        sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
    else:
        sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
        
    # c. Ekran Kartına (veya İşlemciye) Yolla
    x_tensor = torch.tensor([sayisal_dizi], dtype=torch.long).to(device)
    
    # d. Yapay Zekadan Kararı Al
    with torch.no_grad():
        cikis = model(x_tensor)
        tahmin = torch.argmax(cikis, dim=1).item()
        olasilik = torch.softmax(cikis, dim=1)[0][tahmin].item() * 100
        
    durum = "Olumlu" if tahmin == 1 else "Olumsuz"
    
    # e. Müşteriye JSON formatında şık bir cevap dön
    return {
        "orijinal_metin": istek.metin,
        "yapay_zeka_karari": durum,
        "eminlik_orani_yuzde": round(olasilik, 2)
    }