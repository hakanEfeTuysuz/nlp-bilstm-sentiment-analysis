"""Ortak metin temizleme modülü.

Eğitim, değerlendirme ve API aynı fonksiyonu kullanır; böylece eğitimdeki ve
servisteki temizleme birbirinden ayrışmaz.

Sürümler:
    v1: Mevcut imdb_lstm_modeli.pth ile uyumlu, ESKİ davranış. Apostroflu
        kısaltmalar ("isn't") tek kelimeye ("isnt") dönüşür, noktalama silinir.
    v2: Kısaltmalar açılır ("isn't" -> "is not"), "!" ve "?" ayrı token olarak
        korunur. Yeni sözlük ve yeniden eğitim gerektirir.

Not: Model İngilizce IMDb verisiyle eğitildiği için sadece a-z karakterleri
korunur. Türkçe karakter desteği, çok dilli bir model gerektirir.
"""
import re

_HTML = re.compile(r"<[^>]+>")
_BOSLUK = re.compile(r"\s+")

# Sıra önemlidir: özel biçimler (won't, can't) genel "n't" kuralından önce gelir.
_KISALTMALAR = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]


def temizle_v1(metin: str) -> str:
    """Eski davranış: mevcut .pth modeliyle uyumlu."""
    metin = metin.lower()
    metin = _HTML.sub(" ", metin)
    metin = re.sub(r"[^a-z\s]", "", metin)
    return _BOSLUK.sub(" ", metin).strip()


def temizle_v2(metin: str) -> str:
    """Yeni davranış: kısaltmaları açar, ! ve ? işaretlerini token olarak tutar."""
    metin = metin.lower().replace("\u2019", "'")
    metin = _HTML.sub(" ", metin)
    for desen, yerine in _KISALTMALAR:
        metin = desen.sub(yerine, metin)
    metin = re.sub(r"([!?])", r" \1 ", metin)
    metin = re.sub(r"[^a-z!?\s]", "", metin)
    return _BOSLUK.sub(" ", metin).strip()


def temizle(metin: str, surum: str = "v1") -> str:
    if surum == "v1":
        return temizle_v1(metin)
    if surum == "v2":
        return temizle_v2(metin)
    raise ValueError(f"Bilinmeyen temizleme sürümü: {surum!r}")