import torch
import csv
from torch.utils.data import Dataset, DataLoader

print("--- NLP Adım 7: Collate Function ile Tam Entegre Boru Hattı ---\n")

# 1. Sözlük (Vocabulary) Oluşturma (Gerçek projelerde de önden bir kez taranarak oluşturulur)
dosya_adi = "film_yorumlari_veri_seti.csv"
sozluk = {"<PAD>": 0, "<UNK>": 1}
index = 2

# Dosyayı hafifçe tarayıp sadece kelime dağarcığını çıkarıyoruz
with open(dosya_adi, mode='r', encoding='utf-8') as f:
    okuyucu = csv.reader(f)
    next(okuyucu)
    for satir in okuyucu:
        for kelime in satir[0].split():
            if kelime not in sozluk:
                sozluk[kelime] = index
                index += 1

print(f"1. Sözlük Hazır (Toplam {len(sozluk)} Eşsiz Kelime)")

# 2. Dataset Sınıfımız (Bir önceki adımın aynısı)
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
        return self.veriler[idx] # Halen metin ve etiket döndürüyor

# 3. MİMARİNİN KALBİ: Collate Function (Paketleyici İşçi)
SABIT_UZUNLUK = 6 # Bu sefer cümle uzunluğunu 6'ya sabitledik

def veri_paketleyici(gelen_batch):
    # gelen_batch: DataLoader'dan gelen 16 adet (metin, etiket) ikilisi
    islenmis_metinler = []
    etiketler = []
    
    for metin, etiket in gelen_batch:
        # Metni sayılara çevir (Tokenization)
        sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in metin.split()]
        
        # Padding İşlemi (Cümleleri 6 kelimeye sabitleme)
        if len(sayisal_dizi) < SABIT_UZUNLUK:
            sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
        else:
            sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
            
        islenmis_metinler.append(sayisal_dizi)
        etiketler.append(etiket)
        
    # Her şeyi PyTorch Tensor'una çevirip GPU'ya hazır hale getiriyoruz
    return torch.tensor(islenmis_metinler, dtype=torch.long), torch.tensor(etiketler, dtype=torch.long)

# 4. Sistemi Ateşleme
veri_seti = FilmYorumlariDataset(dosya_adi)

# DataLoader'a collate_fn parametresini vererek paketleyici işçimizi sisteme atıyoruz
veri_kargosu = DataLoader(dataset=veri_seti, batch_size=16, shuffle=True, collate_fn=veri_paketleyici)

print("\n2. Kargo Bandı (DataLoader) Testi Başlıyor:\n")
for batch_no, (X_matrisi, Y_etiketleri) in enumerate(veri_kargosu):
    print(f"--- Paket No: {batch_no + 1} ---")
    print(f"X (Girdi) Matris Boyutu: {X_matrisi.shape} -> (16 Cümle, Her Biri 6 Kelime)")
    print(f"Y (Etiket) Matris Boyutu: {Y_etiketleri.shape} -> (16 Etiket)")
    
    print("\nİlk 3 Cümlenin GPU'ya Giden Sayısal Matrisi:")
    print(X_matrisi[:3]) # İlk 3 satırı yazdır
    
    break # Sadece ilk paketi görmek için döngüyü kırıyoruz