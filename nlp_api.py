from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
import torch

from src.model import LSTMDuyguModeli
from src.temizleme import temizle
from src.veri import kodla

print("--- Sistem Başlatılıyor: NLP API Motoru Isınıyor ---")

app = FastAPI(
    title="IMDb NLP Yapay Zeka API",
    description="BiLSTM mimarisi ile eğitilmiş, metinlerin duygusunu analiz eden web servisi.",
    version="2.0.0"
)

# Tarayıcıların API'ye erişmesine izin veren güvenlik kapısı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Her yerden gelen isteklere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. YAYINLANAN MODEL (src.yayinla ile seçilip yayin/ klasörüne kopyalanır)
YAYIN = Path("yayin")
meta = json.loads((YAYIN / "meta.json").read_text(encoding="utf-8"))
sozluk = json.loads((YAYIN / "sozluk.json").read_text(encoding="utf-8"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. MODELİ KURMA VE YÜKLEME
model = LSTMDuyguModeli(
    len(sozluk), meta["vektor_boyutu"], meta["gizli_katman"],
    paketle=meta.get("paketle", False),
).to(device)
model.load_state_dict(torch.load(YAYIN / "model.pth", map_location=device, weights_only=True))
model.eval()  # Eğitimi kapat, sadece hizmet ver!

print(f"-> Model yüklendi: {meta['ad']} | temizleme: {meta['temizleme']} | cihaz: {device.type}")
print("-> API istek bekliyor...")


# 3. GİRİŞ ŞABLONU (Kullanıcıdan ne bekliyoruz?)
class YorumIstegi(BaseModel):
    metin: str


# 4. ENDPOINT (Kullanıcının İstek Atacağı URL Kapısı)
@app.post("/analiz-et")
def duyguyu_analiz_et(istek: YorumIstegi):
    # a. Metni eğitimdeki aynı sürümle temizle
    temiz_metin = temizle(istek.metin, surum=meta["temizleme"])

    # b. Sayılara çevir, sabit uzunluğa getir, gerçek uzunluğu da al
    sayisal_dizi, uzunluk = kodla(temiz_metin, sozluk, meta["sabit_uzunluk"])

    # c. Ekran kartına (veya işlemciye) yolla
    x_tensor = torch.tensor([sayisal_dizi], dtype=torch.long).to(device)
    uzunluklar = torch.tensor([uzunluk], dtype=torch.long)

    # d. Yapay zekadan kararı al
    with torch.no_grad():
        cikis = model(x_tensor, uzunluklar)
        tahmin = torch.argmax(cikis, dim=1).item()
        olasilik = torch.softmax(cikis, dim=1)[0][tahmin].item() * 100

    durum = "Olumlu" if tahmin == 1 else "Olumsuz"

    # e. Müşteriye JSON formatında cevap dön
    return {
        "orijinal_metin": istek.metin,
        "yapay_zeka_karari": durum,
        "eminlik_orani_yuzde": round(olasilik, 2),
        "model": meta["ad"],
    }