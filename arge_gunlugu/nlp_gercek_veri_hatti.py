import torch
import re
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

print("--- NLP Adım 12: Gerçek Veri Boru Hattı (256 Uzunluk) ---\n")

print("1. Veri Yükleniyor ve Sözlük İnşa Ediliyor...")
imdb_verisi = load_dataset("stanfordnlp/imdb")

def metni_temizle(ham_metin):
    metin = ham_metin.lower()
    metin = re.sub(r'<[^>]+>', ' ', metin)
    metin = re.sub(r'[^a-z\s]', '', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()
    return metin

# Hızlı Sözlük İnşası
kelime_sayaci = Counter()
for yorum in imdb_verisi['train']['text']:
    kelime_sayaci.update(metni_temizle(yorum).split())

sozluk = {"<PAD>": 0, "<UNK>": 1}
index = 2
for kelime, _ in kelime_sayaci.most_common(10000):
    sozluk[kelime] = index
    index += 1

print(f"-> Sözlük Hazır! (Boyut: {len(sozluk)})\n")


# 2. HUGGING FACE İÇİN CUSTOM DATASET
class IMDbDataset(Dataset):
    def __init__(self, hf_dataset):
        # Hugging Face'in ayırdığı veriyi (örneğin sadece 'train' kısmı) doğrudan alıyoruz
        self.veriler = hf_dataset
        
    def __len__(self):
        return len(self.veriler)
        
    def __getitem__(self, idx):
        # O anki satırın metnini ve etiketini çekip temizleyerek döndürüyoruz
        metin = self.veriler[idx]['text']
        etiket = self.veriler[idx]['label']
        return metni_temizle(metin), etiket


# 3. KUSURSUZ PAKETLEYİCİ (Collate Function)
SABIT_UZUNLUK = 256

def veri_paketleyici(gelen_batch):
    islenmis_metinler = []
    etiketler = []
    
    for metin, etiket in gelen_batch:
        # Metni sayılara çevir
        sayisal_dizi = [sozluk.get(kelime, sozluk["<UNK>"]) for kelime in metin.split()]
        
        # Uzunluk Kontrolü (Padding ve Truncation)
        if len(sayisal_dizi) < SABIT_UZUNLUK:
            # Kısaysa sonunu 0 (<PAD>) ile doldur
            sayisal_dizi.extend([sozluk["<PAD>"]] * (SABIT_UZUNLUK - len(sayisal_dizi)))
        else:
            # Uzunsa 256. kelimeden sonrasını kırp
            sayisal_dizi = sayisal_dizi[:SABIT_UZUNLUK]
            
        islenmis_metinler.append(sayisal_dizi)
        etiketler.append(etiket)
        
    return torch.tensor(islenmis_metinler, dtype=torch.long), torch.tensor(etiketler, dtype=torch.long)


# 4. SİSTEMİ ATEŞLEME (Kargo Bandı)
# Dataset sınıfımıza Hugging Face'in sadece 'train' (eğitim) verisini veriyoruz
egitim_seti = IMDbDataset(imdb_verisi['train'])

# Ekran kartını boğmamak için verileri 32'şerli paketler halinde yollayacağız
egitim_kargosu = DataLoader(dataset=egitim_seti, batch_size=32, shuffle=True, collate_fn=veri_paketleyici)

print("2. Kargo Bandı (DataLoader) Test Ediliyor...\n")
for X_batch, Y_batch in egitim_kargosu:
    print("--- İlk Paket (Batch) Başarıyla GPU'ya Hazırlandı ---")
    print(f"X (Girdi) Matris Boyutu : {X_batch.shape} -> (32 Yorum, Her Biri 256 Kelime/ID)")
    print(f"Y (Etiket) Matris Boyutu: {Y_batch.shape} -> (32 Adet 0 veya 1)")
    
    print("\nMatrisin İçinden Bir Kesit (İlk Yorumun İlk 15 Kelime ID'si):")
    print(X_batch[0][:15])
    break