"""İkili sınıflandırma metrikleri (saf Python, ek kütüphane gerekmez).

Pozitif sınıf = 1 = "Olumlu". Kesinlik/duyarlılık/F1 "Olumlu" sınıfı içindir;
makro F1 iki sınıfın F1 ortalamasıdır.
"""


def _bol(pay, payda):
    return pay / payda if payda else 0.0


def metrikler(etiketler, tahminler):
    """Etiket ve tahmin listelerinden metrik sözlüğü döndürür (yüzdeler 0-100)."""
    if len(etiketler) != len(tahminler):
        raise ValueError("Etiket ve tahmin sayıları eşit olmalı")
    tp = sum(1 for e, t in zip(etiketler, tahminler) if e == 1 and t == 1)
    tn = sum(1 for e, t in zip(etiketler, tahminler) if e == 0 and t == 0)
    fp = sum(1 for e, t in zip(etiketler, tahminler) if e == 0 and t == 1)
    fn = sum(1 for e, t in zip(etiketler, tahminler) if e == 1 and t == 0)

    kesinlik = _bol(tp, tp + fp)
    duyarlilik = _bol(tp, tp + fn)
    f1_olumlu = _bol(2 * kesinlik * duyarlilik, kesinlik + duyarlilik)

    kesinlik_neg = _bol(tn, tn + fn)
    duyarlilik_neg = _bol(tn, tn + fp)
    f1_olumsuz = _bol(2 * kesinlik_neg * duyarlilik_neg, kesinlik_neg + duyarlilik_neg)

    return {
        "dogruluk": 100 * _bol(tp + tn, len(etiketler)),
        "kesinlik": 100 * kesinlik,
        "duyarlilik": 100 * duyarlilik,
        "f1": 100 * f1_olumlu,
        "makro_f1": 100 * (f1_olumlu + f1_olumsuz) / 2,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }