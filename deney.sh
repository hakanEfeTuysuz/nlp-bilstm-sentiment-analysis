#!/bin/bash
# Her temizleme sürümünü 3 farklı tohumla eğitir, özet satırlarını yazdırır.
# Tam çıktı runs/<ad>.log dosyasına kaydedilir.
mkdir -p runs
for s in 1 2 3; do
  for v in v1 v2; do
    ad=bilstm_${v}_s${s}
    echo "=== $v seed $s ==="
    python3 -m src.egitim --ad $ad --temizleme $v --epoch 20 --tohum $s > runs/$ad.log 2>&1 \
      || { echo "HATA:"; tail -n 15 runs/$ad.log; exit 1; }
    grep -E "en iyi\)|IMDb test|dogrudan|olumsuzun|kisaltma|GENEL|Olumlu tahmin" runs/$ad.log
  done
done