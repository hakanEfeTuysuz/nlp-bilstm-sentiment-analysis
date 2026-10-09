"""Tekrarlanabilir BiLSTM eğitimi (IMDb).

Kullanım (proje kök klasöründen):
    python3 -m src.egitim --ad bilstm_v2 --temizleme v2 --epoch 10
    python3 -m src.egitim --ad bilstm_v2_pack --temizleme v2 --paketle

Çıktılar:
    modeller/<ad>.pth         en iyi doğrulama epoch'unun ağırlıkları
    modeller/<ad>_sozluk.json sözlük
    modeller/<ad>_meta.json   temizleme sürümü ve model ayarları
    sonuclar/<ad>.json        test ve negation sonuçları

Doğrulama (validation) için eğitim verisinin %10'u ayrılır, epoch seçimi buna
göre yapılır. IMDb test seti yalnızca en sonda, bir kez kullanılır.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

from src.degerlendir import dogruluk_hesapla, negation_degerlendir
from src.model import LSTMDuyguModeli
from src.veri import kodla_toplu, sozluk_olustur, temizle_toplu

VEKTOR_BOYUTU = 64
GIZLI_KATMAN = 64
GRADYAN_SINIRI = 1.0


def tohum_ayarla(tohum):
    random.seed(tohum)
    torch.manual_seed(tohum)
    torch.cuda.manual_seed_all(tohum)


def yukleyici_olustur(X, L, y, batch, karistir):
    return DataLoader(TensorDataset(X, L, y), batch_size=batch, shuffle=karistir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad", required=True, help="Çıktı dosyalarının adı, örn. bilstm_v2")
    ap.add_argument("--temizleme", default="v2", choices=["v1", "v2"])
    ap.add_argument("--paketle", action="store_true",
                    help="LSTM'e gerçek uzunlukları ver (pack_padded_sequence)")
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--tohum", type=int, default=42)
    a = ap.parse_args()

    tohum_ayarla(a.tohum)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Cihaz: {cihaz.type.upper()} | temizleme: {a.temizleme} | "
          f"paketle: {a.paketle} | epoch: {a.epoch}")

    print("-> Veri yükleniyor ve temizleniyor...")
    veri = load_dataset("stanfordnlp/imdb")
    temiz_egitim = temizle_toplu(veri["train"]["text"], a.temizleme)
    etiket_egitim = veri["train"]["label"]
    temiz_test = temizle_toplu(veri["test"]["text"], a.temizleme)
    etiket_test = torch.tensor(veri["test"]["label"], dtype=torch.long)

    # %90 eğitim / %10 doğrulama
    n = len(temiz_egitim)
    uretec = torch.Generator().manual_seed(a.tohum)
    sira = torch.randperm(n, generator=uretec).tolist()
    n_val = n // 10
    val_idx, tr_idx = sira[:n_val], sira[n_val:]

    tr_metin = [temiz_egitim[i] for i in tr_idx]
    val_metin = [temiz_egitim[i] for i in val_idx]
    y_tr = torch.tensor([etiket_egitim[i] for i in tr_idx], dtype=torch.long)
    y_val = torch.tensor([etiket_egitim[i] for i in val_idx], dtype=torch.long)

    # Sözlük yalnızca eğitim kısmından kurulur (doğrulama/test sızıntısı olmasın)
    sozluk = sozluk_olustur(tr_metin)
    print(f"-> Sözlük boyutu: {len(sozluk)} | eğitim: {len(tr_metin)} | doğrulama: {len(val_metin)}")

    X_tr, L_tr = kodla_toplu(tr_metin, sozluk)
    X_val, L_val = kodla_toplu(val_metin, sozluk)
    X_te, L_te = kodla_toplu(temiz_test, sozluk)

    tr_yuk = yukleyici_olustur(X_tr, L_tr, y_tr, a.batch, True)
    val_yuk = yukleyici_olustur(X_val, L_val, y_val, 256, False)
    te_yuk = yukleyici_olustur(X_te, L_te, etiket_test, 256, False)

    model = LSTMDuyguModeli(len(sozluk), VEKTOR_BOYUTU, GIZLI_KATMAN, paketle=a.paketle).to(cihaz)
    kayip_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=a.lr)

    en_iyi_val, en_iyi_epoch, en_iyi_durum = -1.0, 0, None
    baslangic = time.time()
    for epoch in range(1, a.epoch + 1):
        model.train()
        toplam_kayip, dogru, toplam = 0.0, 0, 0
        for X, L, y in tr_yuk:
            X, y = X.to(cihaz), y.to(cihaz)
            cikis = model(X, L)
            kayip = kayip_fn(cikis, y)
            opt.zero_grad()
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADYAN_SINIRI)
            opt.step()
            toplam_kayip += kayip.item()
            dogru += (cikis.argmax(dim=1) == y).sum().item()
            toplam += y.size(0)

        val = dogruluk_hesapla(model, val_yuk, cihaz)
        isaret = ""
        if val > en_iyi_val:
            en_iyi_val, en_iyi_epoch = val, epoch
            en_iyi_durum = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            isaret = "  <- en iyi"
        print(
            f"Epoch {epoch:2d}/{a.epoch} | kayıp {toplam_kayip / len(tr_yuk):.4f} | "
            f"eğitim %{100 * dogru / toplam:.2f} | doğrulama %{val:.2f}{isaret}"
        )
    print(f"-> Eğitim süresi: {time.time() - baslangic:.0f} sn")

    # En iyi doğrulama epoch'unu geri yükle ve kaydet
    model.load_state_dict(en_iyi_durum)
    Path("modeller").mkdir(exist_ok=True)
    Path("sonuclar").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"modeller/{a.ad}.pth")
    Path(f"modeller/{a.ad}_sozluk.json").write_text(json.dumps(sozluk), encoding="utf-8")
    Path(f"modeller/{a.ad}_meta.json").write_text(json.dumps({
        "ad": a.ad,
        "temizleme": a.temizleme,
        "paketle": a.paketle,
        "vektor_boyutu": VEKTOR_BOYUTU,
        "gizli_katman": GIZLI_KATMAN,
        "sabit_uzunluk": 256,
        "en_iyi_epoch": en_iyi_epoch,
    }, indent=2), encoding="utf-8")

    # Test seti: yalnızca burada, bir kez
    test_dogruluk = dogruluk_hesapla(model, te_yuk, cihaz)
    print(f"\n-> IMDb test doğruluğu (25.000 yorum): %{test_dogruluk:.2f} (en iyi epoch: {en_iyi_epoch})")

    negation = negation_degerlendir(model, sozluk, a.temizleme, cihaz)
    Path(f"sonuclar/{a.ad}.json").write_text(json.dumps({
        "ad": a.ad,
        "temizleme": a.temizleme,
        "paketle": a.paketle,
        "en_iyi_epoch": en_iyi_epoch,
        "dogrulama_dogrulugu": round(en_iyi_val, 2),
        "test_dogrulugu": round(test_dogruluk, 2),
        "negation": negation,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> Sonuçlar kaydedildi: sonuclar/{a.ad}.json")


if __name__ == "__main__":
    main()