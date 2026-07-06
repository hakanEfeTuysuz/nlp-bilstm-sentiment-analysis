import torch
import csv
from torch.utils.data import Dataset, DataLoader

print("--- NLP Adım 6: PyTorch Dataset ve DataLoader Mimari Kurulumu ---\n")

# 1. GERÇEK DÜNYA SİMÜLASYONU: Diske Örnek Bir Veri Dosyası (CSV) Oluşturma
dosya_adi = "film_yorumlari_veri_seti.csv"
ornek_veriler = [
    ["bu film gerçekten harika ve mükemmel", 1],
    ["berbat bir film zaman kaybı", 0],
    ["film fena değil izlenir", 1],
    ["harika bir senaryo oyunculuk süper", 1],
    ["berbat ve çok kötü senaryo", 0],
    ["bu gerçekten kötü bir yapım", 0],
    ["hayatımda izlediğim en iyi film", 1],
    ["tamamen rezalet bir iş", 0],
] * 10  # Veriyi 80 satıra çıkarmak için çoğaltıyoruz

with open(dosya_adi, mode='w', newline='', encoding='utf-8') as dosya:
    yazici = csv.writer(dosya)
    yazici.writerow(["Yorum", "Etiket"]) # Başlıklar
    yazici.writerows(ornek_veriler)

print(f"1. Sistem: '{dosya_adi}' başarıyla diske kaydedildi (Toplam 80 Satır).\n")


# 2. DATASET MİMARİSİ (Depocu)
# PyTorch'un standart Dataset sınıfından miras alıyoruz
class FilmYorumlariDataset(Dataset):
    def __init__(self, csv_dosyasi):
        # Dosyayı baştan sona okuyup sadece RAM'de liste olarak tutuyoruz
        # (Gerçek çok büyük projelerde bu kısım satır satır okunacak şekilde yazılır)
        self.veriler = []
        with open(csv_dosyasi, mode='r', encoding='utf-8') as f:
            okuyucu = csv.reader(f)
            next(okuyucu) # Başlık satırını (Yorum, Etiket) atla
            for satir in okuyucu:
                self.veriler.append((satir[0], int(satir[1])))
                
    # PyTorch'un bilmesi gereken zorunlu kural 1: Toplam kaç veri var?
    def __len__(self):
        return len(self.veriler)

    # PyTorch'un bilmesi gereken zorunlu kural 2: Belirli bir indeksteki veriyi nasıl alırım?
    def __getitem__(self, idx):
        metin, etiket = self.veriler[idx]
        
        # NORMALDE BURADA METNİ SAYILARA (ID'lere) ÇEVİRİRİZ. 
        # Şimdilik mimariyi görmek için sadece metni döndürüyoruz.
        return metin, etiket


# 3. MİMARİYİ ATEŞLEME ZAMANI
# Depocumuzu çağırıyoruz
orijinal_veri_seti = FilmYorumlariDataset(dosya_adi)
print(f"2. Dataset Hazır: İçeride toplam {len(orijinal_veri_seti)} adet veri bulunuyor.\n")

# Kargo Bandını (DataLoader) kuruyoruz
# batch_size=16: Ekran kartına verileri 80'er 80'er değil, 16'şar 16'şar yolla.
# shuffle=True: Ağırlıkları ezberlememesi (overfitting) için her epoch'ta verileri karıştır.
veri_kargosu = DataLoader(dataset=orijinal_veri_seti, batch_size=16, shuffle=True)

# 4. SİSTEM TESTİ: Kargo bandından ilk paketi alalım
print("3. Kargo Bandı (DataLoader) Testi:\n")
for batch_no, (metin_paketi, etiket_paketi) in enumerate(veri_kargosu):
    print(f"--- Paket (Batch) No: {batch_no + 1} ---")
    print(f"Bu paketteki eleman sayısı: {len(metin_paketi)}")
    print(f"İlk 2 Metin: {metin_paketi[:2]}")
    print(f"Etiketler (Tensor): {etiket_paketi}\n")
    
    # Döngüyü kırıyoruz ki sadece ilk paketi görelim
    break