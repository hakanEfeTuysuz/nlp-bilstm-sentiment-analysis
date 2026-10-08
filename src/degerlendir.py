"""Değerlendirme yardımcıları: doğruluk hesabı ve olumsuzluk (negation) test seti."""
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.veri import kodla_toplu, temizle_toplu

NEGATION_DOSYASI = Path("tests/negation_seti.json")


def dogruluk_hesapla(model, yukleyici, cihaz):
    """DataLoader (X, uzunluklar, etiket) üzerinde doğruluk yüzdesi döndürür."""
    model.eval()
    dogru = toplam = 0
    with torch.no_grad():
        for X, L, y in yukleyici:
            tahmin = model(X.to(cihaz), L).argmax(dim=1).cpu()
            dogru += (tahmin == y).sum().item()
            toplam += y.size(0)
    return 100 * dogru / toplam


def tahmin_et(model, sozluk, metinler, surum, cihaz):
    """Ham metin listesi için (tahminler, eminlik_yüzdeleri) döndürür."""
    X, L = kodla_toplu(temizle_toplu(metinler, surum), sozluk)
    model.eval()
    with torch.no_grad():
        olasiliklar = torch.softmax(model(X.to(cihaz), L), dim=1)
    tahminler = olasiliklar.argmax(dim=1)
    eminlikler = olasiliklar.max(dim=1).values * 100
    return tahminler.cpu().tolist(), eminlikler.cpu().tolist()


def negation_degerlendir(model, sozluk, surum, cihaz, dosya=NEGATION_DOSYASI, yazdir=True):
    """Negation test setini ölçer. Baseline JSON dosyasıyla aynı şemayı döndürür."""
    ornekler = json.loads(Path(dosya).read_text(encoding="utf-8"))
    tahminler, eminlikler = tahmin_et(
        model, sozluk, [o["metin"] for o in ornekler], surum, cihaz
    )

    dogru = defaultdict(int)
    toplam = defaultdict(int)
    hatalar = []
    for o, tahmin, eminlik in zip(ornekler, tahminler, eminlikler):
        toplam[o["kategori"]] += 1
        if tahmin == o["etiket"]:
            dogru[o["kategori"]] += 1
        else:
            hatalar.append({
                "metin": o["metin"],
                "beklenen": "Olumlu" if o["etiket"] == 1 else "Olumsuz",
                "tahmin": "Olumlu" if tahmin == 1 else "Olumsuz",
                "eminlik": round(eminlik, 2),
                "kategori": o["kategori"],
            })

    n = len(ornekler)
    toplam_dogru = sum(dogru.values())
    sonuc = {
        "genel_dogruluk": round(100 * toplam_dogru / n, 2),
        "olumlu_tahmin_orani": round(100 * sum(tahminler) / n, 2),
        "kategoriler": {k: {"dogru": dogru[k], "toplam": toplam[k]} for k in toplam},
        "hatalar": hatalar,
    }

    if yazdir:
        print("\n=== Olumsuzluk Test Seti ===")
        for k in toplam:
            print(f"{k:22s} {dogru[k]:2d}/{toplam[k]:2d}  (%{100 * dogru[k] / toplam[k]:.1f})")
        print("-" * 45)
        print(f"{'GENEL':22s} {toplam_dogru:2d}/{n:2d}  (%{sonuc['genel_dogruluk']:.1f})")
        print(f"Olumlu tahmin oranı: %{sonuc['olumlu_tahmin_orani']:.1f} (gerçek oran %50)")
    return sonuc