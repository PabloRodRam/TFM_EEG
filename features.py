"""
Extracción de features y cómputo de espectrogramas STFT.

Tres conjuntos de features usados en el TFM:

  - **v1 (gamma, 12 feats/ch)**: ``extract_features_window``
    temporal + frecuencial sobre señal gamma filtrada.
  - **LOSO (Flujo, 13 feats/ch)**: ``extract_features_vector``
    temporal + espectral (Welch PSD) sobre señal 0.5-40 Hz.
  - **v4 (broadband, 345 feats)**: ``extract_svm_features_v4``
    estadísticas por banda sobre espectrogramas STFT.

Espectrogramas:
  - ``compute_spectrogram_batch`` (v1, n_fft=256, hop=32)
  - ``compute_spectrogram_v4``    (v4, n_fft=256, hop=128)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew

from .config import (
    DEVICE,
    FREQ_BANDS,
    FS,
    N_CHANNELS,
    STFT_HOP,
    STFT_HOP_V4,
    STFT_NFFT,
    STFT_NFFT_V4,
    WIN_SAMPLES,
)


# ================================================================
# Features v1 — gamma (12 feats / canal = 276 total)
# ================================================================

def extract_features_window(x_win: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extrae 12 features clásicos de una ventana ``(n_samples, n_ch)``.

    Features por canal:
      - Temporal: mean, std, skewness, kurtosis, RMS, line length
      - Frecuencial: band power en 4 sub-bandas gamma (30-50, 50-70,
        70-90, 90-127 Hz), entropía espectral, frecuencia pico

    Returns: vector 1-D ``(n_ch × 12,)``
    """
    n_samples, n_ch = x_win.shape
    feats: list[float] = []

    for ch in range(n_ch):
        sig = x_win[:, ch]

        # Temporal
        f_mean = np.mean(sig)
        f_std = np.std(sig)
        f_skew = skew(sig)
        f_kurt = sp_kurtosis(sig)
        f_rms = np.sqrt(np.mean(sig**2))
        f_ll = np.sum(np.abs(np.diff(sig)))

        # Frecuencial (FFT)
        fft_vals = np.abs(np.fft.rfft(sig))
        fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

        bands = [(30, 50), (50, 70), (70, 90), (90, 127)]
        bp: list[float] = []
        total_power = np.sum(fft_vals**2) + 1e-12
        for lo, hi in bands:
            mask_b = (fft_freqs >= lo) & (fft_freqs <= hi)
            bp.append(float(np.sum(fft_vals[mask_b] ** 2) / total_power))

        # Entropía espectral
        psd = fft_vals**2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        psd_norm = psd_norm[psd_norm > 0]
        f_se = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

        # Frecuencia pico
        f_peak = float(fft_freqs[np.argmax(fft_vals)])

        feats.extend(
            [f_mean, f_std, f_skew, f_kurt, f_rms, f_ll] + bp + [f_se, f_peak]
        )

    return np.array(feats, dtype=np.float32)


def extract_features_batch(X: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extrae features v1 para un batch ``(n, win_samples, n_ch)``."""
    return np.array([extract_features_window(X[i], fs) for i in range(len(X))])


# ================================================================
# Features LOSO — Flujo (13 feats / canal)
# ================================================================

def extract_features_vector(
    window_data: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """Extrae 13 features por canal vía Welch PSD (LOSO cross-subject SVM).

    Input: ``(n_channels, n_timepoints)``

    Features por canal:
      - Temporal: mean, std, skew, kurtosis
      - Forma: line length, energy, ZCR
      - Espectral: 5 band powers (delta, theta, alpha, beta, gamma), entropía

    Returns: vector 1-D ``(n_channels × 13,)``
    """
    n_channels, n_samples = window_data.shape

    # Diff para line length
    diff_sig = np.diff(window_data, axis=1)

    # Welch PSD
    freqs, psd = welch(window_data, fs=fs, nperseg=n_samples // 2, axis=1)

    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

    # Stats temporales
    mean_v = np.mean(window_data, axis=1)
    std_v = np.std(window_data, axis=1)
    import scipy.stats
    skew_v = scipy.stats.skew(window_data, axis=1)
    kurt_v = scipy.stats.kurtosis(window_data, axis=1)

    # Métricas de forma
    line_len_v = np.sum(np.abs(diff_sig), axis=1)
    energy_v = np.sum(window_data**2, axis=1)
    zcr_v = np.sum(
        np.abs(np.diff(np.sign(window_data), axis=1)), axis=1,
    ) / (2 * n_samples)

    # Espectrales
    total_power_v = np.sum(psd, axis=1)

    band_powers = []
    for fmin, fmax in bands:
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        bp = np.sum(psd[:, idx_band], axis=1)
        band_powers.append(bp)
    band_powers_v = np.stack(band_powers, axis=1)  # (n_ch, 5)

    # Entropía espectral
    psd_norm = psd / (total_power_v[:, None] + 1e-12)
    spectral_entropy_v = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)

    # Concatenar: (n_ch, 13)
    features_per_channel = np.column_stack([
        mean_v, std_v, skew_v, kurt_v,
        line_len_v, energy_v, zcr_v,
        band_powers_v,
        spectral_entropy_v,
    ])

    return features_per_channel.flatten().astype(np.float32)


# ================================================================
# Features v4 — SVM broadband (345 feats = 23 ch × 5 bandas × 3 stats)
# ================================================================

def extract_svm_features_v4(
    X_spec: np.ndarray,
    fs: int = FS,
    n_fft: int = STFT_NFFT_V4,
) -> np.ndarray:
    """Extrae features estadísticas de espectrogramas para SVM v4.

    Para cada canal y banda (delta, theta, alpha, beta, gamma):
      - Media de potencia log
      - Desviación típica
      - Kurtosis

    Input  : ``(n, 23, n_freq, n_time)``
    Output : ``(n, 345)``
    """
    n_samples, n_channels, n_freq, n_time = X_spec.shape
    freq_resolution = fs / n_fft  # Hz por bin

    features: list[np.ndarray] = []
    for _band_name, (f_lo, f_hi) in FREQ_BANDS.items():
        bin_lo = max(0, int(f_lo / freq_resolution))
        bin_hi = min(n_freq, int(f_hi / freq_resolution) + 1)
        band_data = X_spec[:, :, bin_lo:bin_hi, :]  # (n, ch, bins, time)

        flat = band_data.reshape(n_samples, n_channels, -1)  # (n, ch, bins*time)

        mean_feat = flat.mean(axis=2)
        std_feat = flat.std(axis=2)
        kurt_feat = np.apply_along_axis(
            lambda x: sp_kurtosis(x, fisher=True) if len(x) > 3 else 0.0,
            axis=2,
            arr=flat,
        )

        features.extend([mean_feat, std_feat, kurt_feat])

    X_features = np.column_stack(features)  # (n, 23*5*3 = 345)
    return X_features.astype(np.float32)


# ================================================================
# Espectrogramas STFT
# ================================================================

def compute_spectrogram_batch(
    X_raw: np.ndarray,
    n_fft: int = STFT_NFFT,
    hop_length: int = STFT_HOP,
    device: torch.device = DEVICE,
    chunk_size: int = 256,
) -> np.ndarray:
    """STFT v1 (gamma): ``(n, 1280, 23) → (n, 23, n_freq, n_time)``.

    Parámetros: n_fft=256, hop=32.
    """
    n_samples, win_samples, n_channels = X_raw.shape
    window = torch.hann_window(n_fft).to(device)

    # Dimensiones de salida
    test_sig = torch.zeros(1, win_samples, device=device)
    test_out = torch.stft(
        test_sig, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window, center=True, return_complex=True,
    )
    n_freq, n_time = test_out.shape[1], test_out.shape[2]
    del test_sig, test_out

    X_spec = np.zeros(
        (n_samples, n_channels, n_freq, n_time), dtype=np.float32,
    )

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch = torch.tensor(X_raw[start:end], dtype=torch.float32, device=device)
        flat = batch.permute(0, 2, 1).reshape(-1, win_samples)

        Zxx = torch.stft(
            flat, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, window=window, center=True, return_complex=True,
        )
        mag = torch.log1p(torch.abs(Zxx))

        X_spec[start:end] = (
            mag.reshape(end - start, n_channels, n_freq, n_time).cpu().numpy()
        )
        del batch, flat, Zxx, mag

    torch.cuda.empty_cache()
    return X_spec


def compute_spectrogram_v4(
    X_raw: np.ndarray,
    n_fft: int = STFT_NFFT_V4,
    hop_length: int = STFT_HOP_V4,
    device: torch.device = DEVICE,
    chunk_size: int = 256,
) -> np.ndarray:
    """STFT v4 (broadband): ``(n, 1280, 23) → (n, 23, n_freq, n_time)``.

    Parámetros: n_fft=256, hop=128.  Con estos parámetros:
    n_freq=129, n_time≈11.
    """
    n_samples, win_samples, n_channels = X_raw.shape
    window = torch.hann_window(n_fft).to(device)

    test_sig = torch.zeros(1, win_samples, device=device)
    test_out = torch.stft(
        test_sig, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window, center=True, return_complex=True,
    )
    n_freq, n_time = test_out.shape[1], test_out.shape[2]
    del test_sig, test_out

    X_spec = np.zeros(
        (n_samples, n_channels, n_freq, n_time), dtype=np.float32,
    )

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch = torch.tensor(X_raw[start:end], dtype=torch.float32, device=device)
        flat = batch.permute(0, 2, 1).reshape(-1, win_samples)

        Zxx = torch.stft(
            flat, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, window=window, center=True, return_complex=True,
        )
        mag = torch.log1p(torch.abs(Zxx))

        X_spec[start:end] = (
            mag.reshape(end - start, n_channels, n_freq, n_time).cpu().numpy()
        )
        del batch, flat, Zxx, mag

    torch.cuda.empty_cache()
    return X_spec
