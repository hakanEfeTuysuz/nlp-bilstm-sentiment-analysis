"""Olumsuzluk (negation) test seti değerlendirmesi.

Şu an canlı API'nin kullandığı model ve temizleme hattını ölçer
(nlp_api.duyguyu_analiz_et), yani kullanıcının gerçekte gördüğü davranışı.

Kullanım:
    python3 degerlendir_negation.py [sonuc_dosyasi.json]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from nlp_api import duyguyu_analiz_et, YorumIstegi

TEST_DOSYASI = Path("tests/negation_seti.json")
SONUC_DOSYASI = Path(sys.argv[1] if len(sys.argv) > 1 else "sonuclar/negation_sonuc.json")

ornekler = json.loads(TEST_DOSYASI.read_text(encoding="utf-8"))

dogru = defaultdict(int)
toplam = defaultdict(int)
hatalar = []
olumlu_tahmin = 0

for o in ornekler:
    cevap = duyguyu_analiz_et(YorumIstegi(metin=o["metin"]))
    tahmin = 1 if cevap["yapay_zeka_karari"] == "Olumlu" else 0
    olumlu_tahmin += tahmin
    toplam[o["kategori"]] += 1
    if tahmin == o["etiket"]:
        dogru[o["kategori"]] += 1
    else:
        hatalar.append({
            "metin": o["metin"],
            "beklenen": "Olumlu" if o["etiket"] == 1 else "Olumsuz",
            "tahmin": cevap["yapay_zeka_karari"],
            "eminlik": cevap["eminlik_orani_yuzde"],
            "kategori": o["kategori"],
        })

n = len(ornekler)
toplam_dogru = sum(dogru.values())

print("\n=== Olumsuzluk Test Seti Sonuçları ===")
for kategori in toplam:
    print(f"{kategori:22s} {dogru[kategori]:2d}/{toplam[kategori]:2d}  (%{100 * dogru[kategori] / toplam[kategori]:.1f})")
print("-" * 45)
print(f"{'GENEL':22s} {toplam_dogru:2d}/{n:2d}  (%{100 * toplam_dogru / n:.1f})")
print(f"Olumlu tahmin oranı: %{100 * olumlu_tahmin / n:.1f} (gerçek oran %50)")

print(f"\n=== Hatalar ({len(hatalar)}) ===")
for h in hatalar:
    print(f"[{h['kategori']}] {h['metin']}")
    print(f"    beklenen: {h['beklenen']} | tahmin: {h['tahmin']} (%{h['eminlik']})")

SONUC_DOSYASI.parent.mkdir(parents=True, exist_ok=True)
SONUC_DOSYASI.write_text(json.dumps({
    "genel_dogruluk": round(100 * toplam_dogru / n, 2),
    "olumlu_tahmin_orani": round(100 * olumlu_tahmin / n, 2),
    "kategoriler": {k: {"dogru": dogru[k], "toplam": toplam[k]} for k in toplam},
    "hatalar": hatalar,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSonuçlar kaydedildi: {SONUC_DOSYASI}")