"""Veri hazırlama: sözlük oluşturma ve metinleri sabit uzunlukta kodlama."""
from collections import Counter

import torch

from src.temizleme import temizle

PAD = "<PAD>"
UNK = "<UNK>"
SABIT_UZUNLUK = 256
SOZLUK_KELIME_SAYISI = 10000


def temizle_toplu(metinler, surum="v1"):
    """Ham metin listesini seçilen sürümle temizler."""
    return [temizle(m, surum=surum) for m in metinler]


def sozluk_olustur(temiz_metinler, en_cok=SOZLUK_KELIME_SAYISI):
    """En sık geçen `en_cok` kelimeden sözlük kurar. 0=<PAD>, 1=<UNK>."""
    sayac = Counter()
    for metin in temiz_metinler:
        sayac.update(metin.split())
    sozluk = {PAD: 0, UNK: 1}
    for sira, (kelime, _) in enumerate(sayac.most_common(en_cok), start=2):
        sozluk[kelime] = sira
    return sozluk


def kodla(temiz_metin, sozluk, uzunluk=SABIT_UZUNLUK):
    """Tek metni sayı dizisine çevirir. (dizi, gerçek_uzunluk) döndürür."""
    idler = [sozluk.get(k, sozluk[UNK]) for k in temiz_metin.split()][:uzunluk]
    gercek = max(len(idler), 1)  # boş metinde bile uzunluk en az 1
    idler = idler + [sozluk[PAD]] * (uzunluk - len(idler))
    return idler, gercek


def kodla_toplu(temiz_metinler, sozluk, uzunluk=SABIT_UZUNLUK):
    """Metin listesini (X, uzunluklar) tensörlerine çevirir."""
    dizi, uzunluklar = [], []
    for metin in temiz_metinler:
        idler, gercek = kodla(metin, sozluk, uzunluk)
        dizi.append(idler)
        uzunluklar.append(gercek)
    return (
        torch.tensor(dizi, dtype=torch.long),
        torch.tensor(uzunluklar, dtype=torch.long),
    )