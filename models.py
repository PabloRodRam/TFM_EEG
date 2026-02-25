"""
Arquitecturas de redes neuronales para predicción de crisis epilépticas.

Modelos incluidos (en orden cronológico del TFM):

  - **SeizureLSTM** (v1): LSTM unidireccional, 1 capa, 128 hidden.
  - **SeizureBiLSTM** (v3): BiLSTM, 2 capas, 128 hidden/dir.
  - **SeizureCNNLSTM** (v4): CNN 2-bloque + BiLSTM 2-capas.
  - **SeizureTransformer** (v1/v4): CNN 3-bloque + Transformer 2-capas 8-head
    con positional encoding sinusoidal (CrossEntropy, 2 clases).
  - **ImprovedCNNTransformer** (Flujo): CNN 3-bloque piramidal + Transformer
    3-capas 4-head con positional embedding aprendido (BCE, 1 logit).
  - **FocalLoss**: pérdida focal basada en BCE para desequilibrio de clases.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_CHANNELS


# ================================================================
# SeizureLSTM (v1)
# ================================================================

class SeizureLSTM(nn.Module):
    """LSTM unidireccional para predicción de crisis (Fase 1 v1).

    Entrada : ``(batch, 1280, 23)``  — señal gamma cruda normalizada.
    Salida  : ``(batch, 2)``         — logits (CrossEntropyLoss).
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        hidden_size: int = 128,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_hidden))


# ================================================================
# SeizureBiLSTM (v3)
# ================================================================

class SeizureBiLSTM(nn.Module):
    """BiLSTM bidireccional, 2 capas (Fase 1/2 v3).

    Entrada : ``(batch, 1280, 23)``
    Salida  : ``(batch, 2)``
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        hidden_size: int = 128,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_hidden))


# ================================================================
# SeizureCNNLSTM (v4 — Fase 3)
# ================================================================

class SeizureCNNLSTM(nn.Module):
    """CNN-LSTM híbrido para espectrogramas broadband (Fase 3 v4).

    La CNN colapsa la dimensión de frecuencia; el BiLSTM procesa la
    secuencia temporal resultante.

    Entrada : ``(batch, 23, n_freq, n_time)``
    Salida  : ``(batch, 2)``
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_freq: int = 129,
        d_cnn: int = 64,
        lstm_hidden: int = 128,
        n_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            # Bloque 1
            nn.Conv2d(n_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),  # freq /4
            # Bloque 2
            nn.Conv2d(64, d_cnn, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(d_cnn),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # colapsar freq → 1
        )
        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)          # (B, d_cnn, 1, T)
        h = h.squeeze(2)         # (B, d_cnn, T)
        h = h.permute(0, 2, 1)   # (B, T, d_cnn)
        lstm_out, _ = self.lstm(h)
        last = lstm_out[:, -1, :]
        return self.fc(self.dropout(last))


# ================================================================
# SeizureTransformer (v1 / v4 — usado en ambas fases)
# ================================================================

class SeizureTransformer(nn.Module):
    """CNN-Transformer con positional encoding sinusoidal (2 clases).

    Entrada : ``(batch, 23, n_freq, n_time)`` — espectrograma log-magnitud.
    Salida  : ``(batch, 2)``                  — logits (CrossEntropyLoss).
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_freq: int = 129,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(128, d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # colapsar freq → 1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Classification head
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

        self._pe_cache: torch.Tensor | None = None

    def _get_positional_encoding(
        self, seq_len: int, device: torch.device,
    ) -> torch.Tensor:
        if self._pe_cache is not None and self._pe_cache.size(1) >= seq_len:
            return self._pe_cache[:, :seq_len, :]
        pe = torch.zeros(1, seq_len, self.d_model, device=device)
        position = torch.arange(
            0, seq_len, dtype=torch.float32, device=device,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self._pe_cache = pe
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)                   # (B, d_model, 1, T)
        h = h.squeeze(2).permute(0, 2, 1) # (B, T, d_model)
        h = h + self._get_positional_encoding(h.size(1), h.device)
        h = self.transformer(h)
        h = h.mean(dim=1)                 # GAP
        h = self.dropout_layer(h)
        return self.fc(h)


# ================================================================
# ImprovedCNNTransformer (Flujo — Fase 1 LOSO CNN-Transformer)
# ================================================================

class ImprovedCNNTransformer(nn.Module):
    """CNN-Transformer piramidal con positional embedding aprendido (1 logit).

    Usado en el flujo LOSO de Fase 1 (FocalLoss + BCE).

    Entrada : ``(batch, n_channels, n_freq, n_time)``
    Salida  : ``(batch, 1)``  — logit (usar ``BCEWithLogitsLoss``).
    """

    def __init__(
        self,
        num_channels: int = 18,
        num_freqs: int = 65,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN encoder — estructura piramidal en frecuencia
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))  # F/2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))  # F/4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))  # F/8

        final_freq_dim = num_freqs // 8
        self.cnn_output_dim = 128 * final_freq_dim

        self.proj = nn.Linear(self.cnn_output_dim, d_model)
        self.dropout_emb = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Positional embedding aprendido
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F_dim, T = x.shape

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten: (B, 128, F', T) → (B, T, 128*F')
        x = x.permute(0, 3, 1, 2)     # (B, T, C, F')
        x = x.reshape(B, T, -1)        # (B, T, cnn_output_dim)

        x = self.proj(x)               # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout_emb(x)

        x = self.transformer(x)

        # Global Average Pooling
        x = x.mean(dim=1)              # (B, d_model)
        return self.fc(x)              # (B, 1)


# ================================================================
# FocalLoss
# ================================================================

class FocalLoss(nn.Module):
    """Focal Loss basado en BCE (Lin et al. 2017).

    Se usa con modelos de salida 1 logit (``ImprovedCNNTransformer``).
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none",
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()
