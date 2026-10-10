"""Kayıtlı modelleri IMDb testi ve iki negation setinde ölçüp rapor üretir.

Kullanım:
    python3 -m src.rapor [ad_oneki]            # varsayılan: bilstm_v
    python3 -m src.rapor bilstm_v --readme     # README'deki sonuç bölümünü de günceller

Çıktı: sonuclar/rapor.md
Eğitim gerekmez: modeller/ klasöründeki modeller ve (varsa) kökteki eski
baseline (imdb_lstm_modeli.pth + imdb_sozluk.json) yüklenir.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from src.degerlendir import tahmin_et
from src.degerlendir_kayitli import modeli_yukle, ort_std
from src.metrik import metrikler
from src.model import LSTMDuyguModeli
from src.veri import kodla_toplu, temizle_toplu

NEGATION_SETLERI = {
    "neg1": ("Olumsuzluk test seti 1 (60 cümle)", "tests/negation_seti.json"),
    "neg2": ("Olumsuzluk test seti 2 (44 cümle, farklı kalıplar)", "tests/negation_seti_2.json"),
}
BASLANGIC = "<!-- SONUC_TABLOSU_BASLA -->"
BITIS = "<!-- SONUC_TABLOSU_BITIS -->"
SIRA = ["baseline", "bilstm_v1", "bilstm_v2", "bilstm_v2_pack", "bilstm_v2_pack_sablon"]
METRIKLER = ("dogruluk", "kesinlik", "duyarlilik", "f1", "makro_f1")


def toplu_tahmin(model, sozluk, temiz_metinler, cihaz, batch=512):
    """Temizlenmiş metinleri parçalar halinde modele verir (GPU belleği için)."""
    X, L = kodla_toplu(temiz_metinler, sozluk)
    tahminler = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            cikis = model(X[i:i + batch].to(cihaz), L[i:i + batch])
            tahminler.extend(cikis.argmax(dim=1).cpu().tolist())
    return tahminler


def baseline_yukle(cihaz):
    """README'deki eski model: imdb_lstm_modeli.pth + imdb_sozluk.json, temizleme v1."""
    sozluk = json.loads(Path("imdb_sozluk.json").read_text(encoding="utf-8"))
    model = LSTMDuyguModeli(len(sozluk), 64, 64)
    model.load_state_dict(torch.load("imdb_lstm_modeli.pth", map_location=cihaz, weights_only=True))
    return model.to(cihaz).eval(), sozluk, "v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onek", nargs="?", default="bilstm_v")
    ap.add_argument("--readme", action="store_true")
    a = ap.parse_args()

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-> IMDb test seti yükleniyor...")
    test = load_dataset("stanfordnlp/imdb")["test"]
    test_etiket = list(test["label"])
    temiz_onbellek = {}

    def test_temiz(surum):
        if surum not in temiz_onbellek:
            temiz_onbellek[surum] = temizle_toplu(test["text"], surum)
        return temiz_onbellek[surum]

    neg_verileri = {}
    for anahtar, (_, dosya) in NEGATION_SETLERI.items():
        ornekler = json.loads(Path(dosya).read_text(encoding="utf-8"))
        neg_verileri[anahtar] = ([o["metin"] for o in ornekler], [o["etiket"] for o in ornekler])

    def olc(model, sozluk, surum):
        sonuc = {"imdb": metrikler(test_etiket, toplu_tahmin(model, sozluk, test_temiz(surum), cihaz))}
        for anahtar, (metinler, etiketler) in neg_verileri.items():
            tahmin, _ = tahmin_et(model, sozluk, metinler, surum, cihaz)
            sonuc[anahtar] = metrikler(etiketler, tahmin)
        return sonuc

    gruplar = defaultdict(list)
    if Path("imdb_lstm_modeli.pth").exists() and Path("imdb_sozluk.json").exists():
        print("-> baseline (imdb_lstm_modeli.pth)")
        model, sozluk, surum = baseline_yukle(cihaz)
        gruplar["baseline"].append(olc(model, sozluk, surum))

    for meta_yolu in sorted(Path("modeller").glob(f"{a.onek}*_meta.json")):
        model, sozluk, meta = modeli_yukle(meta_yolu, cihaz)
        print(f"-> {meta['ad']}")
        grup = re.sub(r"_s\d+$", "", meta["ad"])
        gruplar[grup].append(olc(model, sozluk, meta["temizleme"]))

    if not gruplar:
        raise SystemExit("Ölçülecek model bulunamadı.")

    sirali = sorted(gruplar, key=lambda g: (SIRA.index(g) if g in SIRA else len(SIRA), g))
    gruplar = {g: gruplar[g] for g in sirali}
    bolum = rapor_olustur(gruplar)

    Path("sonuclar").mkdir(exist_ok=True)
    Path("sonuclar/rapor.md").write_text(bolum + "\n", encoding="utf-8")
    print("\n" + bolum)
    print("\n-> Kaydedildi: sonuclar/rapor.md")
    if a.readme:
        readme_guncelle(bolum)
        print("-> README.md güncellendi")


def metrik_tablosu(gruplar, anahtar):
    satirlar = [
        "| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |",
        "|---|---|---|---|---|---|---|",
    ]
    for grup, kosular in gruplar.items():
        hucreler = [ort_std([k[anahtar][m] for k in kosular]) for m in METRIKLER]
        satirlar.append(f"| {grup} | {len(kosular)} | " + " | ".join(hucreler) + " |")
    return "\n".join(satirlar)


def karisiklik_tablosu(gruplar):
    satirlar = ["| Model | Test seti | TN | FP | FN | TP |", "|---|---|---|---|---|---|"]
    adlar = {"imdb": "IMDb test", "neg1": "Negation 1", "neg2": "Negation 2"}
    for grup, kosular in gruplar.items():
        for anahtar, ad in adlar.items():
            o = {k: statistics.mean(r[anahtar][k] for r in kosular) for k in ("tn", "fp", "fn", "tp")}
            satirlar.append(
                f"| {grup} | {ad} | {o['tn']:.0f} | {o['fp']:.0f} | {o['fn']:.0f} | {o['tp']:.0f} |"
            )
    return "\n".join(satirlar)


def rapor_olustur(gruplar):
    parcalar = [
        "## 📈 Deney Sonuçları",
        "",
        "Değerler, koşular (farklı rastgele tohumlar) arasında ortalama ± standart sapmadır. "
        "Kesinlik, duyarlılık ve F1 \"Olumlu\" sınıfı içindir. `baseline`, projenin ilk "
        "sürümündeki model (`imdb_lstm_modeli.pth`) olup tek koşudur. Bu tablo "
        "`python3 -m src.rapor --readme` ile otomatik üretilir.",
        "",
        "### IMDb test seti (25.000 yorum)",
        "",
        metrik_tablosu(gruplar, "imdb"),
    ]
    for anahtar, (baslik, _) in NEGATION_SETLERI.items():
        parcalar += ["", f"### {baslik}", "", metrik_tablosu(gruplar, anahtar)]
    parcalar += [
        "",
        "### Karışıklık matrisleri (koşu başına ortalama)",
        "",
        "TN: olumsuzu olumsuz bildi, FP: olumsuzu olumlu sandı, "
        "FN: olumluyu olumsuz sandı, TP: olumluyu olumlu bildi.",
        "",
        karisiklik_tablosu(gruplar),
    ]
    return "\n".join(parcalar)


def readme_guncelle(bolum):
    yol = Path("README.md")
    metin = yol.read_text(encoding="utf-8")
    blok = f"{BASLANGIC}\n{bolum}\n{BITIS}"
    if BASLANGIC in metin and BITIS in metin:
        metin = re.sub(
            re.escape(BASLANGIC) + r".*?" + re.escape(BITIS),
            lambda _: blok, metin, flags=re.S,
        )
    else:
        eslesme = re.search(r"^## .*Bilinen Sınırlar ve Zayıf Yönler", metin, flags=re.M)
        if not eslesme:
            raise SystemExit("README'de 'Bilinen Sınırlar ve Zayıf Yönler' başlığı bulunamadı.")
        metin = metin[:eslesme.start()] + blok + "\n\n" + metin[eslesme.start():]
    yol.write_text(metin, encoding="utf-8")


if __name__ == "__main__":
    main()