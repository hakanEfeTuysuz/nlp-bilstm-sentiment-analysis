"""Olumsuzluk (negation) için dengeli şablon eğitim verisi.

Kısa, dilbilgisi doğru cümleler üretir. Etiket = sıfatın kutbu, olumsuzlama
varsa tersine çevrilir ("not excellent" -> olumsuz, "not tedious" -> olumlu).

Tasarım kuralları:
  * Sıfatlar, tests/negation_seti.json içindeki hiçbir kelimeyle çakışmaz
    (test cümleleri ezberlenemesin). `python3 -m src.sablon_veri` bunu kontrol eder.
  * Olumsuzlama, olumlu ve olumsuz sıfatlarla eşit sayıda görülür; yani "not"
    kelimesi tek başına bir etiket ipucu olmaz.

Kullanım:
    python3 -m src.sablon_veri
"""
import json
import random
import re
from pathlib import Path

OLUMLU_SIFATLAR = [
    "excellent", "superb", "delightful", "charming", "gripping", "hilarious",
    "touching", "impressive", "clever", "engaging", "stunning", "moving",
    "memorable", "outstanding", "thrilling", "refreshing", "original",
    "powerful", "riveting", "splendid", "lovely", "pleasant", "appealing",
    "polished", "inspiring",
]
OLUMSUZ_SIFATLAR = [
    "dreadful", "tedious", "annoying", "pointless", "lame", "silly",
    "forgettable", "mediocre", "dreary", "clumsy", "weak", "messy",
    "shallow", "tiresome", "dismal", "laughable", "bland", "confusing",
    "irritating", "lifeless", "stale", "sloppy", "painful", "unwatchable",
    "overlong",
]

# (cümle başı, cümle ortası)
VARLIKLAR = [
    ("The movie", "the movie"), ("This film", "this film"),
    ("The acting", "the acting"), ("The story", "the story"),
    ("The plot", "the plot"), ("The ending", "the ending"),
    ("The script", "the script"), ("The dialogue", "the dialogue"),
    ("The soundtrack", "the soundtrack"), ("The pacing", "the pacing"),
    ("The cast", "the cast"), ("The direction", "the direction"),
]

# {V}: cümle başı varlık, {v}: cümle ortası varlık, {A}: sıfat
DUZ_KALIPLAR = [
    "{V} is {A}.",
    "{V} was really {A}.",
    "I found {v} {A}.",
    "Overall, {v} is {A}.",
]
OLUMSUZ_KALIPLAR = [
    "{V} is not {A}.",
    "{V} was not {A}.",
    "{V} isn't {A}.",
    "{V} wasn't {A}.",
    "{V} is not {A} at all.",
    "{V} was never {A}.",
    "I did not find {v} {A}.",
    "I didn't find {v} {A}.",
    "Overall, {v} is not {A}.",
]


def sablon_uret(adet, tohum=42):
    """(metinler, etiketler) döndürür. etiket: 1=olumlu, 0=olumsuz. Dengeli üretir."""
    rng = random.Random(tohum)
    metinler, etiketler = [], []
    # Dört kombinasyon sırayla döner: (kutup, olumsuzlama) -> eşit sayıda
    kombinasyonlar = [(1, False), (0, False), (1, True), (0, True)]
    for i in range(adet):
        kutup, olumsuzlama = kombinasyonlar[i % 4]
        sifat = rng.choice(OLUMLU_SIFATLAR if kutup == 1 else OLUMSUZ_SIFATLAR)
        kalip = rng.choice(OLUMSUZ_KALIPLAR if olumsuzlama else DUZ_KALIPLAR)
        bas, orta = rng.choice(VARLIKLAR)
        metinler.append(kalip.format(V=bas, v=orta, A=sifat))
        etiketler.append(kutup ^ 1 if olumsuzlama else kutup)
    return metinler, etiketler


def cakisma_kontrol(test_dosyasi="tests/negation_seti.json"):
    """Şablon sıfatları ile test seti kelimeleri arasındaki çakışmayı döndürür."""
    ornekler = json.loads(Path(test_dosyasi).read_text(encoding="utf-8"))
    test_kelimeleri = set()
    for o in ornekler:
        test_kelimeleri.update(re.findall(r"[a-z]+", o["metin"].lower()))
    return (set(OLUMLU_SIFATLAR) | set(OLUMSUZ_SIFATLAR)) & test_kelimeleri


if __name__ == "__main__":
    cakisma = cakisma_kontrol()
    assert not cakisma, f"Test setiyle çakışan sıfatlar: {cakisma}"
    assert not set(OLUMLU_SIFATLAR) & set(OLUMSUZ_SIFATLAR), "Kutuplar çakışıyor"
    metinler, etiketler = sablon_uret(4000)
    olumsuzlu = [("not" in m or "n't" in m or "never" in m) for m in metinler]
    print(f"Üretilen cümle: {len(metinler)} | olumlu etiket: {sum(etiketler)}")
    print(f"Olumsuzlamalı: {sum(olumsuzlu)} | olumsuzlamalı ve olumlu etiketli: "
          f"{sum(1 for o, e in zip(olumsuzlu, etiketler) if o and e == 1)}")
    print("Test seti ile sıfat çakışması: yok")
    print("\nÖrnekler:")
    for m, e in list(zip(metinler, etiketler))[:8]:
        print(f"  [{'Olumlu ' if e else 'Olumsuz'}] {m}")