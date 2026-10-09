"""Sözlük ve kırpma kayıplarını ölçer (eğitim verisi, temizleme v2).

Kullanım:
    python3 -m src.veri_analizi
"""
import statistics

from datasets import load_dataset

from src.veri import SABIT_UZUNLUK, sozluk_olustur, temizle_toplu

veri = load_dataset("stanfordnlp/imdb")
metinler = temizle_toplu(veri["train"]["text"], "v2")
sozluk = sozluk_olustur(metinler)

uzunluklar = [len(m.split()) for m in metinler]
toplam_token = sum(uzunluklar)
kirpilan = sum(max(0, u - SABIT_UZUNLUK) for u in uzunluklar)
uzun_yorum = sum(1 for u in uzunluklar if u > SABIT_UZUNLUK)
sozluk_disi = sum(1 for m in metinler for k in m.split() if k not in sozluk)

print(f"Yorum sayısı: {len(metinler)}")
print(f"Medyan uzunluk: {statistics.median(uzunluklar):.0f} kelime | ortalama: {statistics.mean(uzunluklar):.0f}")
print(f"{SABIT_UZUNLUK} kelimeden uzun yorum: %{100 * uzun_yorum / len(metinler):.1f}")
print(f"Kırpmayla atılan kelime oranı: %{100 * kirpilan / toplam_token:.1f}")
print(f"Sözlük dışı (<UNK>) kelime oranı: %{100 * sozluk_disi / toplam_token:.1f}")