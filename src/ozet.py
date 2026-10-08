"""sonuclar/*.json dosyalarını tohum ortalamasıyla özetler (Markdown tablosu).

Dosya adı `<ad>_s<tohum>.json` biçimindeyse aynı <ad>'a sahip koşular gruplanır.

Kullanım:
    python3 -m src.ozet
    python3 -m src.ozet bilstm_v   # yalnızca adı bununla başlayan gruplar
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OLUMSUZLUK_KATEGORILERI = ["dogrudan_olumsuzluk", "olumsuzun_olumsuzu", "kisaltma"]


def olumsuzluk_yuzdesi(negation):
    dogru = sum(negation["kategoriler"][k]["dogru"] for k in OLUMSUZLUK_KATEGORILERI)
    toplam = sum(negation["kategoriler"][k]["toplam"] for k in OLUMSUZLUK_KATEGORILERI)
    return 100 * dogru / toplam


def ort_std(degerler):
    ort = statistics.mean(degerler)
    std = statistics.stdev(degerler) if len(degerler) > 1 else 0.0
    return f"{ort:.1f} ± {std:.1f}"


def main():
    filtre = sys.argv[1] if len(sys.argv) > 1 else ""
    gruplar = defaultdict(list)
    for yol in sorted(Path("sonuclar").glob("*.json")):
        veri = json.loads(yol.read_text(encoding="utf-8"))
        if "test_dogrulugu" not in veri:  # örn. negation_baseline.json
            continue
        grup = re.sub(r"_s\d+$", "", yol.stem)
        if grup.startswith(filtre):
            gruplar[grup].append(veri)

    print("| Model | Koşu | IMDb test % | Negation genel % | Olumsuzluk cümleleri % | Olumlu tahmin % |")
    print("|---|---|---|---|---|---|")
    for grup, kosular in gruplar.items():
        print(
            f"| {grup} | {len(kosular)} "
            f"| {ort_std([k['test_dogrulugu'] for k in kosular])} "
            f"| {ort_std([k['negation']['genel_dogruluk'] for k in kosular])} "
            f"| {ort_std([olumsuzluk_yuzdesi(k['negation']) for k in kosular])} "
            f"| {ort_std([k['negation']['olumlu_tahmin_orani'] for k in kosular])} |"
        )


if __name__ == "__main__":
    main()