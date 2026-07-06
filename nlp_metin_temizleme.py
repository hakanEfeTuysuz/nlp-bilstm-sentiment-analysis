import re
from datasets import load_dataset

print("--- NLP Adım 10: Regex ile Metin Temizleme ve Normalize Etme ---\n")

# Veriyi (önbellekten) hızlıca tekrar yüklüyoruz
imdb_verisi = load_dataset("stanfordnlp/imdb")
ornek_metin = imdb_verisi['train'][0]['text']

def metni_temizle(ham_metin):
    # 1. Bütün metni küçük harfe çevir (I AM -> i am)
    metin = ham_metin.lower()
    
    # 2. HTML etiketlerini (<br />, <i>, vb.) boşlukla değiştir
    # <[^>]+> : '<' ile başla, '>' görene kadar ne varsa seç demek
    metin = re.sub(r'<[^>]+>', ' ', metin)
    
    # 3. Noktalama işaretlerini ve rakamları sil (Sadece a'dan z'ye harfler kalsın)
    # [^a-z\s] : 'a' ile 'z' arası harfler ve boşluk (\s) HARİCİNDEKİ her şeyi sil
    metin = re.sub(r'[^a-z\s]', '', metin)
    
    # 4. Fazladan oluşan yan yana boşlukları tek boşluğa indir
    metin = re.sub(r'\s+', ' ', metin).strip()
    
    return metin

# Temizleme motorunu çalıştır
temizlenmis_metin = metni_temizle(ornek_metin)

print("1. Orijinal Kirli Metin (İlk 200 Karakter):")
print(ornek_metin[:200] + "...\n")

print("2. Temizlenmiş Saf Metin (İlk 200 Karakter):")
print(temizlenmis_metin[:200] + "...\n")

print("3. Kelime Sayısı Karşılaştırması:")
print(f"Eski (Kirli) Kelime Sayısı : {len(ornek_metin.split())}")
print(f"Yeni (Temiz) Kelime Sayısı : {len(temizlenmis_metin.split())}")