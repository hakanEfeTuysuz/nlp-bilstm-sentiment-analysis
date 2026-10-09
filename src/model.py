"""BiLSTM duygu analizi modeli.

Mimari, mevcut imdb_lstm_modeli.pth ile birebir uyumludur (aynı katman
adları ve boyutlar). `paketle=True` verilirse LSTM yalnızca gerçek token
uzunluğu kadar okur (pack_padded_sequence); boş (PAD) tokenler son hafızayı
bozmaz. Ağırlık adları ve boyutları paketle'den bağımsız aynıdır.
"""
import torch
import torch.nn as nn


class LSTMDuyguModeli(nn.Module):
    def __init__(self, sozluk_boyutu, vektor_boyutu=64, gizli_katman=64, paketle=False):
        super().__init__()
        self.paketle = paketle
        self.embedding = nn.Embedding(
            num_embeddings=sozluk_boyutu, embedding_dim=vektor_boyutu, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=vektor_boyutu, hidden_size=gizli_katman,
            batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(gizli_katman * 2, 2)

    def forward(self, x, uzunluklar=None):
        gomumler = self.embedding(x)
        if self.paketle and uzunluklar is not None:
            paket = nn.utils.rnn.pack_padded_sequence(
                gomumler, uzunluklar.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(paket)
        else:
            _, (hidden, _) = self.lstm(gomumler)
        son_hafiza = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(son_hafiza))