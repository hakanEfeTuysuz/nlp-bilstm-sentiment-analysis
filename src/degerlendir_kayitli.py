"""Kaydedilmiş modelleri bir negation test dosyasında ölçer (yeniden eğitim gerekmez).

Kullanım:
    python3 -m src.degerlendir_kayitli <ad_oneki> [test_dosyasi]
Örnek:
    python3 -m src.degerlendir_kayitli bilstm_v tests/negation_seti_2.json

modeller/<ad>_meta.json dosyası olan, adı <ad_oneki> ile başlayan her model ölçülür.
`_s<tohum>` son eki dışında aynı adı taşıyanlar gruplanır ve ortalaması alınır.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

from src.degerlendir import negation_degerlendir
from src.model import LSTMDuyguModeli


def modeli_yukle(meta_yolu, cihaz):
    meta = json.loads(Path(meta_yolu).read_text(encoding="utf-8"))
    ad = meta["ad"]
    sozluk = json.loads(Path(f"modeller/{ad}_sozluk.json").read_text(encoding="utf-8"))
    model = LSTMDuyguModeli(
        len(sozluk), meta["vektor_boyutu"], meta["gizli_katman"],
        paketle=meta.get("paketle", False),
    )
    model.load_state_dict(
        torch.load(f"modeller/{ad}.pth", map_location=cihaz, weights_only=True)
    )
    return model.to(cihaz).eval(), sozluk, meta


def ort_std(degerler):
    ort = statistics.mean(degerler)
    std = statistics.stdev(degerler) if len(degerler) > 1 else 0.0
    return f"{ort:.1f} ± {std:.1f}"


def main():
    if len(sys.argv) < 2:
        sys.exit("Kullanım: python3 -m src.degerlendir_kayitli <ad_oneki> [test_dosyasi]")
    onek = sys.argv[1]
    test_dosyasi = sys.argv[2] if len(sys.argv) > 2 else "tests/negation_seti.json"
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gruplar = defaultdict(list)
    for meta_yolu in sorted(Path("modeller").glob(f"{onek}*_meta.json")):
        model, sozluk, meta = modeli_yukle(meta_yolu, cihaz)
        sonuc = negation_degerlendir(
            model, sozluk, meta["temizleme"], cihaz, dosya=test_dosyasi, yazdir=False
        )
        gruplar[re.sub(r"_s\d+$", "", meta["ad"])].append(sonuc)

    if not gruplar:
        sys.exit(f"modeller/ altında '{onek}' ile başlayan model bulunamadı.")

    kategoriler = list(next(iter(gruplar.values()))[0]["kategoriler"])
    print(f"Test dosyası: {test_dosyasi}\n")
    print("| Model | Koşu | Genel % | " + " | ".join(f"{k} %" for k in kategoriler) + " | Olumlu tahmin % |")
    print("|---|---|---|" + "---|" * len(kategoriler) + "---|")
    for grup, kosular in gruplar.items():
        kategori_hucreleri = []
        for k in kategoriler:
            yuzdeler = [100 * r["kategoriler"][k]["dogru"] / r["kategoriler"][k]["toplam"] for r in kosular]
            kategori_hucreleri.append(ort_std(yuzdeler))
        print(
            f"| {grup} | {len(kosular)} "
            f"| {ort_std([r['genel_dogruluk'] for r in kosular])} | "
            + " | ".join(kategori_hucreleri)
            + f" | {ort_std([r['olumlu_tahmin_orani'] for r in kosular])} |"
        )


if __name__ == "__main__":
    main()