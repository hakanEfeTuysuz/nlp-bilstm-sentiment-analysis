"""Doğrulama doğruluğu en yüksek modeli yayin/ klasörüne kopyalar (API bunu kullanır).

Seçim yalnızca doğrulama (validation) doğruluğuna göre yapılır; negation ve IMDb
test setleri seçimde kullanılmaz.

Kullanım:
    python3 -m src.yayinla <ad_oneki>
Örnek:
    python3 -m src.yayinla bilstm_v2_pack_sablon
"""
import json
import re
import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        sys.exit("Kullanım: python3 -m src.yayinla <ad_oneki>")
    onek = sys.argv[1]

    adaylar = []
    for yol in Path("sonuclar").glob(f"{onek}_s*.json"):
        # "bilstm_v2_pack" öneki "bilstm_v2_pack_sablon_s1" ile karışmasın
        if not re.fullmatch(rf"{re.escape(onek)}_s\d+", yol.stem):
            continue
        veri = json.loads(yol.read_text(encoding="utf-8"))
        adaylar.append((veri["dogrulama_dogrulugu"], veri["ad"], veri["negation"]["genel_dogruluk"]))
    if not adaylar:
        sys.exit(f"sonuclar/ altında '{onek}_s<tohum>.json' bulunamadı.")

    adaylar.sort()
    print("Adaylar (doğrulama doğruluğu, ad):")
    for dogrulama, ad, _ in adaylar:
        print(f"  %{dogrulama:.2f}  {ad}")
    dogrulama, ad, neg1 = adaylar[-1]

    Path("yayin").mkdir(exist_ok=True)
    for kaynak, hedef in [
        (f"modeller/{ad}.pth", "yayin/model.pth"),
        (f"modeller/{ad}_sozluk.json", "yayin/sozluk.json"),
        (f"modeller/{ad}_meta.json", "yayin/meta.json"),
    ]:
        shutil.copyfile(kaynak, hedef)

    print(f"\nSeçilen: {ad} (doğrulama %{dogrulama:.2f})")
    print("yayin/ klasörüne kopyalandı: model.pth, sozluk.json, meta.json")
    print(f"Kontrol için: API ile negation seti 1 genel doğruluğu %{neg1:.2f} çıkmalı.")


if __name__ == "__main__":
    main()