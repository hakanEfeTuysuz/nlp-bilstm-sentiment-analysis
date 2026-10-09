#!/bin/bash
# Kullanım: bash deney.sh <ad_oneki> <temizleme> "<tohumlar>" [ek egitim argumanlari...]
# Örnek:    bash deney.sh bilstm_v2_pack v2 "1 2 3 4 5" --paketle
# Tam çıktı runs/<ad>.log dosyasına kaydedilir.
onek=$1
temizleme=$2
tohumlar=$3
shift 3
mkdir -p runs
for s in $tohumlar; do
  ad=${onek}_s${s}
  echo "=== $ad ==="
  python3 -m src.egitim --ad "$ad" --temizleme "$temizleme" --epoch 20 --tohum "$s" "$@" > "runs/$ad.log" 2>&1 \
    || { echo "HATA:"; tail -n 15 "runs/$ad.log"; exit 1; }
  grep -E "IMDb test|GENEL|Olumlu tahmin" "runs/$ad.log"
done