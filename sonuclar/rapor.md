## 📈 Deney Sonuçları

Değerler, koşular (farklı rastgele tohumlar) arasında ortalama ± standart sapmadır. Kesinlik, duyarlılık ve F1 "Olumlu" sınıfı içindir. `baseline`, projenin ilk sürümündeki model (`imdb_lstm_modeli.pth`) olup tek koşudur. Bu tablo `python3 -m src.rapor --readme` ile otomatik üretilir.

### IMDb test seti (25.000 yorum)

| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |
|---|---|---|---|---|---|---|
| baseline | 1 | 83.7 ± 0.0 | 83.3 ± 0.0 | 84.3 ± 0.0 | 83.8 ± 0.0 | 83.7 ± 0.0 |
| bilstm_v1 | 3 | 83.4 ± 0.2 | 84.5 ± 1.6 | 81.8 ± 2.8 | 83.1 ± 0.6 | 83.4 ± 0.2 |
| bilstm_v2 | 5 | 84.4 ± 0.5 | 83.9 ± 1.5 | 85.2 ± 2.7 | 84.5 ± 0.8 | 84.4 ± 0.5 |
| bilstm_v2_pack | 5 | 84.8 ± 0.4 | 86.5 ± 0.2 | 82.5 ± 0.6 | 84.4 ± 0.4 | 84.8 ± 0.4 |
| bilstm_v2_pack_sablon | 5 | 85.1 ± 0.2 | 85.0 ± 1.1 | 85.2 ± 1.3 | 85.1 ± 0.2 | 85.1 ± 0.2 |
| bilstm_v2_pack_dikkat | 5 | 86.1 ± 0.5 | 85.1 ± 1.9 | 87.7 ± 2.6 | 86.3 ± 0.6 | 86.1 ± 0.5 |
| bilstm_v2_pack_sablon_dikkat | 5 | 86.2 ± 0.6 | 86.3 ± 2.1 | 86.1 ± 4.2 | 86.1 ± 1.1 | 86.2 ± 0.7 |

### Olumsuzluk test seti 1 (60 cümle)

| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |
|---|---|---|---|---|---|---|
| baseline | 1 | 40.0 ± 0.0 | 42.1 ± 0.0 | 53.3 ± 0.0 | 47.1 ± 0.0 | 38.9 ± 0.0 |
| bilstm_v1 | 3 | 57.8 ± 4.2 | 68.6 ± 18.6 | 46.7 ± 26.5 | 50.0 ± 10.9 | 54.4 ± 3.3 |
| bilstm_v2 | 5 | 57.3 ± 4.8 | 63.1 ± 10.6 | 41.3 ± 6.5 | 49.1 ± 2.8 | 55.9 ± 3.8 |
| bilstm_v2_pack | 5 | 56.0 ± 4.2 | 58.6 ± 6.1 | 41.3 ± 3.8 | 48.5 ± 4.4 | 55.0 ± 4.1 |
| bilstm_v2_pack_sablon | 5 | 67.7 ± 1.9 | 75.2 ± 6.1 | 54.0 ± 5.5 | 62.5 ± 1.6 | 66.9 ± 1.5 |
| bilstm_v2_pack_dikkat | 5 | 57.0 ± 2.2 | 61.2 ± 4.4 | 39.3 ± 2.8 | 47.8 ± 2.0 | 55.6 ± 1.9 |
| bilstm_v2_pack_sablon_dikkat | 5 | 74.3 ± 6.9 | 81.3 ± 7.4 | 63.3 ± 12.5 | 70.7 ± 9.1 | 73.9 ± 7.2 |

### Olumsuzluk test seti 2 (44 cümle, farklı kalıplar)

| Model | Koşu | Doğruluk % | Kesinlik % | Duyarlılık % | F1 % | Makro F1 % |
|---|---|---|---|---|---|---|
| baseline | 1 | 52.3 ± 0.0 | 52.6 ± 0.0 | 45.5 ± 0.0 | 48.8 ± 0.0 | 52.0 ± 0.0 |
| bilstm_v1 | 3 | 58.3 ± 3.5 | 64.5 ± 9.5 | 43.9 ± 13.1 | 50.7 ± 5.1 | 56.7 ± 2.2 |
| bilstm_v2 | 5 | 57.7 ± 5.9 | 63.5 ± 10.2 | 39.1 ± 2.5 | 48.2 ± 4.5 | 56.2 ± 5.4 |
| bilstm_v2_pack | 5 | 59.5 ± 3.0 | 64.7 ± 4.3 | 41.8 ± 5.0 | 50.7 ± 4.7 | 58.2 ± 3.3 |
| bilstm_v2_pack_sablon | 5 | 62.3 ± 4.1 | 68.5 ± 8.5 | 47.3 ± 2.5 | 55.7 ± 2.7 | 61.4 ± 3.7 |
| bilstm_v2_pack_dikkat | 5 | 61.8 ± 1.9 | 67.7 ± 3.8 | 45.5 ± 0.0 | 54.4 ± 1.2 | 60.8 ± 1.7 |
| bilstm_v2_pack_sablon_dikkat | 5 | 60.0 ± 3.0 | 64.0 ± 3.9 | 45.5 ± 4.5 | 53.1 ± 4.3 | 59.1 ± 3.3 |

### Karışıklık matrisleri (koşu başına ortalama)

TN: olumsuzu olumsuz bildi, FP: olumsuzu olumlu sandı, FN: olumluyu olumsuz sandı, TP: olumluyu olumlu bildi.

| Model | Test seti | TN | FP | FN | TP |
|---|---|---|---|---|---|
| baseline | IMDb test | 10390 | 2110 | 1966 | 10534 |
| baseline | Negation 1 | 8 | 22 | 14 | 16 |
| baseline | Negation 2 | 13 | 9 | 12 | 10 |
| bilstm_v1 | IMDb test | 10621 | 1879 | 2276 | 10224 |
| bilstm_v1 | Negation 1 | 21 | 9 | 16 | 14 |
| bilstm_v1 | Negation 2 | 16 | 6 | 12 | 10 |
| bilstm_v2 | IMDb test | 10448 | 2052 | 1855 | 10645 |
| bilstm_v2 | Negation 1 | 22 | 8 | 18 | 12 |
| bilstm_v2 | Negation 2 | 17 | 5 | 13 | 9 |
| bilstm_v2_pack | IMDb test | 10889 | 1611 | 2188 | 10312 |
| bilstm_v2_pack | Negation 1 | 21 | 9 | 18 | 12 |
| bilstm_v2_pack | Negation 2 | 17 | 5 | 13 | 9 |
| bilstm_v2_pack_sablon | IMDb test | 10620 | 1880 | 1856 | 10644 |
| bilstm_v2_pack_sablon | Negation 1 | 24 | 6 | 14 | 16 |
| bilstm_v2_pack_sablon | Negation 2 | 17 | 5 | 12 | 10 |
| bilstm_v2_pack_dikkat | IMDb test | 10572 | 1928 | 1541 | 10959 |
| bilstm_v2_pack_dikkat | Negation 1 | 22 | 8 | 18 | 12 |
| bilstm_v2_pack_dikkat | Negation 2 | 17 | 5 | 12 | 10 |
| bilstm_v2_pack_sablon_dikkat | IMDb test | 10776 | 1724 | 1732 | 10768 |
| bilstm_v2_pack_sablon_dikkat | Negation 1 | 26 | 4 | 11 | 19 |
| bilstm_v2_pack_sablon_dikkat | Negation 2 | 16 | 6 | 12 | 10 |
