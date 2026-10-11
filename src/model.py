"""BiLSTM duygu analizi modeli.

Mimari, mevcut imdb_lstm_modeli.pth ile birebir uyumludur (aynı katman
adları ve boyutlar).

paketle=True : LSTM yalnızca gerçek token uzunluğu kadar okur
               (pack_padded_sequence); boş (PAD) tokenler son hafızayı bozmaz.
dikkat=True  : Son gizli durum yerine, tüm zaman adımlarının dikkat ağırlıklı
               ortalaması kullanılır (additive attention). PAD konumları
               maskelenir. Yalnızca bu seçenekle ek katmanlar (dikkat_w,
               dikkat_v) oluşur; eski ağırlık dosyaları etkilenmez.
"""
import torch
import torch.nn as nn


class LSTMDuyguModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu=64, gizli_katman=64,
                 paketle=False, dikkat=False):
        super().__init__()
        self.paketle = paketle
        self.dikkat = dikkat
        self.embedding = nn.Embedding(
            num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=vektor_boyutu, hidden_size=gizli_katman,
            batch_first=True, bidirectional=True,
        )
        if dikkat:
            self.dikkat_w = nn.Linear(gizli_katman * 2, gizli_katman * 2)
            self.dikkat_v = nn.Linear(gizli_katman * 2, 1, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(gizli_katman * 2, 2)

    def forward(self, x, uzunluklar=None, agirlik_dondur=False):
        gomumler = self.embedding(x)
        if self.paketle and uzunluklar is not None:
            paket = nn.utils.rnn.pack_padded_sequence(
                gomumler, uzunluklar.cpu(), batch_first=True, enforce_sorted=False
            )
            cikis, (hidden, _) = self.lstm(paket)
            cikis, _ = nn.utils.rnn.pad_packed_sequence(
                cikis, batch_first=True, total_length=x.size(1)
            )
        else:
            cikis, (hidden, _) = self.lstm(gomumler)

        agirliklar = None
        if self.dikkat:
            adim_sayisi = x.size(1)
            if uzunluklar is not None:
                maske = (
                    torch.arange(adim_sayisi, device=x.device).unsqueeze(0)
                    < uzunluklar.to(x.device).unsqueeze(1)
                )
            else:
                maske = x != 0
                maske[:, 0] = True  # tamamen boş metinde bile en az bir konum açık
            skorlar = self.dikkat_v(torch.tanh(self.dikkat_w(cikis))).squeeze(-1)
            skorlar = skorlar.masked_fill(~maske, float("-inf"))
            agirliklar = torch.softmax(skorlar, dim=1)
            ozet = torch.bmm(agirliklar.unsqueeze(1), cikis).squeeze(1)
        else:
            ozet = torch.cat((hidden[-2], hidden[-1]), dim=1)

        sonuc = self.fc(self.dropout(ozet))
        if agirlik_dondur:
            return sonuc, agirliklar
        return sonuc