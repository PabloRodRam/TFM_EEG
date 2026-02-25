"""
Preprocesamiento de señal EEG.

Incluye filtrado (gamma y broadband), normalización, etiquetado temporal,
segmentación en ventanas, limpieza de artefactos y pipeline completo por
sujeto en dos variantes:

  - **v1 (Fase 1)**: filtro gamma 30-127 Hz + undersampling interictal.
  - **v4 (Fase 3)**: filtro broadband 0.5-127 Hz, sin undersampling
    (usa class weights), genera espectrogramas y features SVM.
"""

from __future__ import annotations

import gc as _gc

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .config import (
    BROADBAND_HI,
    BROADBAND_LO,
    BUTTER_ORDER,
    DEVICE,
    FS,
    GAMMA_HI,
    GAMMA_LO,
    INTER_GAP_POST,
    INTER_GAP_PRE,
    N_CHANNELS,
    POSTICTAL_MIN,
    PREICTAL_MIN,
    STFT_HOP_V4,
    STFT_NFFT_V4,
    WIN_SAMPLES,
    WIN_SEC,
)
from .data import load_subject_continuous


# ================================================================
# Diseño de filtros Butterworth
# ================================================================

def design_gamma_butter(fs: int = FS, order: int = BUTTER_ORDER):
    """Filtro Butterworth pasa-banda gamma (30-127 Hz).  Representación SOS."""
    nyq = fs / 2.0
    low = GAMMA_LO / nyq
    high = min(GAMMA_HI, nyq - 1.0) / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def design_broadband_butter(fs: int = FS, order: int = BUTTER_ORDER):
    """Filtro Butterworth pasa-banda broadband (0.5-127 Hz).  Representación SOS."""
    nyq = fs / 2.0
    low = BROADBAND_LO / nyq
    high = min(BROADBAND_HI, nyq - 1.0) / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


# Pre-calcular coeficientes SOS
GAMMA_SOS = design_gamma_butter()
BROADBAND_SOS = design_broadband_butter()


# ================================================================
# Aplicación de filtros
# ================================================================

def apply_gamma_filter(data: np.ndarray) -> np.ndarray:
    """Aplica filtro Butterworth gamma (30-127 Hz) a cada canal."""
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = sosfiltfilt(GAMMA_SOS, data[:, ch])
    return filtered


def apply_broadband_filter(data: np.ndarray) -> np.ndarray:
    """Aplica filtro Butterworth broadband (0.5-127 Hz) a cada canal."""
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = sosfiltfilt(BROADBAND_SOS, data[:, ch])
    return filtered


# ================================================================
# Normalización
# ================================================================

def normalize_per_channel(data: np.ndarray) -> np.ndarray:
    """Normalización z-score por canal (media = 0, std = 1)."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (data - mean) / std


# ================================================================
# Etiquetado temporal
# ================================================================

def label_segments(
    n_samples: int,
    seizure_abs_list: list[dict],
    ref_ts: float,
    fs: int = FS,
    preictal_min: int = PREICTAL_MIN,
    postictal_min: int = POSTICTAL_MIN,
) -> np.ndarray:
    """Asigna etiqueta a cada muestra:

    -1 = descartar (ictal / indefinido)
     0 = interictal
     1 = preictal
     2 = postictal
    """
    labels = np.full(n_samples, -1, dtype=np.int8)

    sz_intervals = []
    for sz in seizure_abs_list:
        s = int(sz["abs_start"] * fs)
        e = int(sz["abs_end"] * fs)
        sz_intervals.append((s, e))
    sz_intervals.sort()

    total = n_samples

    # Marcar ictal
    for s, e in sz_intervals:
        labels[max(0, s): min(total, e)] = -1

    # Marcar preictal
    pre_samples = preictal_min * 60 * fs
    for s, _e in sz_intervals:
        pre_start = max(0, s - pre_samples)
        labels[pre_start:s] = 1

    # Marcar postictal
    post_samples = postictal_min * 60 * fs
    for _s, e in sz_intervals:
        labels[e: min(total, e + post_samples)] = 2

    # Marcar interictal (>= 5 h post y >= 3 h pre)
    inter_post_samples = INTER_GAP_POST * fs
    inter_pre_samples = INTER_GAP_PRE * fs

    for i in range(len(sz_intervals) + 1):
        range_start = 0 if i == 0 else sz_intervals[i - 1][1] + inter_post_samples
        range_end = total if i == len(sz_intervals) else sz_intervals[i][0] - inter_pre_samples

        if range_start < range_end:
            s_c = max(0, int(range_start))
            e_c = min(total, int(range_end))
            mask = labels[s_c:e_c] == -1
            labels[s_c:e_c][mask] = 0

    return labels


# ================================================================
# Segmentación en ventanas
# ================================================================

def segment_and_label(
    data_filtered: np.ndarray,
    labels: np.ndarray,
    win_samples: int = WIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Segmenta en ventanas de *win_samples* y asigna etiqueta mayoritaria.

    Descarta ventanas con etiqueta -1 (ictal / indefinido).
    """
    n_windows = len(data_filtered) // win_samples
    X = data_filtered[: n_windows * win_samples].reshape(n_windows, win_samples, -1)
    L = labels[: n_windows * win_samples].reshape(n_windows, win_samples)

    win_labels = np.zeros(n_windows, dtype=np.int8)
    keep = np.ones(n_windows, dtype=bool)

    for i in range(n_windows):
        valid = L[i][L[i] >= 0]
        if len(valid) == 0:
            keep[i] = False
            continue
        counts = np.bincount(valid, minlength=3)
        win_labels[i] = counts.argmax()

    return X[keep], win_labels[keep]


# ================================================================
# Limpieza de artefactos EMG
# ================================================================

def remove_emg_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Elimina ventanas cuya amplitud máxima supera el percentil dado."""
    max_amp = np.abs(X).max(axis=(1, 2))
    threshold = np.percentile(max_amp, percentile)
    mask = max_amp <= threshold
    print(
        f"  Artefactos EMG: descartadas {(~mask).sum()} ventanas "
        f"(umbral {threshold:.2e}, percentil {percentile})"
    )
    return X[mask], y[mask]


# ================================================================
# Undersampling interictal (Fase 1 v1)
# ================================================================

def undersample_interictal(
    X: np.ndarray,
    y: np.ndarray,
    target_class: int = 0,
    ref_class: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-muestreo aleatorio de la clase *target* para igualar *ref*."""
    rng = np.random.RandomState(seed)
    n_ref = (y == ref_class).sum()
    idx_target = np.where(y == target_class)[0]
    idx_other = np.where(y != target_class)[0]

    if len(idx_target) > n_ref:
        idx_target = rng.choice(idx_target, size=n_ref, replace=False)

    idx = np.sort(np.concatenate([idx_target, idx_other]))
    print(
        f"  Undersampling interictal: {(y == target_class).sum()} -> "
        f"{len(idx_target)} (igualado a {n_ref} preictal)"
    )
    return X[idx], y[idx]


# ================================================================
# Pipeline completo — v1 (Fase 1, gamma)
# ================================================================

def process_subject(
    subject: str,
    seizures_abs: list[dict],
    binary: bool = True,
) -> dict | None:
    """Pipeline v1 completo para un sujeto:

    1. Carga continua
    2. Filtrado gamma (30-127 Hz)
    3. z-score por canal
    4. Etiquetado (60 min preictal)
    5. Segmentación 5 s
    6. Limpieza artefactos EMG
    7. Undersampling interictal
    """
    print(f"\n{'=' * 60}")
    print(f"Procesando {subject} ({len(seizures_abs)} leading seizures)")
    print(f"{'=' * 60}")

    data, ref_ts = load_subject_continuous(subject)
    if data.size == 0:
        print(f"  No se pudo cargar datos para {subject}")
        return None
    print(
        f"  Cargados {data.shape[0] / FS / 3600:.1f} horas "
        f"({data.shape[0]} muestras, {data.shape[1]} canales)"
    )

    data_gamma = apply_gamma_filter(data)
    data_gamma = normalize_per_channel(data_gamma)

    labels = label_segments(data_gamma.shape[0], seizures_abs, ref_ts, FS)
    X, y = segment_and_label(data_gamma, labels)

    if binary:
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]

    if (y == 1).sum() == 0:
        print(f"  Sin muestras preictales para {subject}")
        return None

    X, y = remove_emg_artifacts(X, y)
    X, y = undersample_interictal(X, y)

    print(
        f"  Dataset final: {X.shape[0]} ventanas, "
        f"preictal={int((y == 1).sum())}, interictal={int((y == 0).sum())}"
    )
    return {"X": X, "y": y}


# ================================================================
# Pipeline completo — v4 (Fase 3, broadband)
# ================================================================

def process_subject_v4(
    subject: str,
    seizures_abs: list[dict],
    binary: bool = True,
) -> dict | None:
    """Pipeline v4 (Fase 3) para un sujeto — *memory-efficient*:

    1. Carga continua
    2. Filtrado broadband (0.5-127 Hz)
    3. z-score por canal
    4. Etiquetado (60 min preictal)
    5. Segmentación 5 s
    6. Limpieza artefactos EMG
    7. **Sin** undersampling (usa class weights)
    8. STFT → espectrogramas log-magnitud
    9. Features estadísticas para SVM
    """
    from .features import compute_spectrogram_v4, extract_svm_features_v4

    print(f"\n{'=' * 60}")
    print(f"[v4] Procesando {subject} ({len(seizures_abs)} leading seizures)")
    print(f"{'=' * 60}")

    data, ref_ts = load_subject_continuous(subject)
    if data.size == 0:
        print(f"  No se pudo cargar datos para {subject}")
        return None
    print(f"  Cargados {data.shape[0] / FS / 3600:.1f} h ({data.shape[1]} canales)")

    data_broad = apply_broadband_filter(data)
    del data
    _gc.collect()

    data_broad = normalize_per_channel(data_broad)

    labels = label_segments(data_broad.shape[0], seizures_abs, ref_ts, FS)
    X_raw, y = segment_and_label(data_broad, labels)
    del data_broad, labels
    _gc.collect()

    if binary:
        mask = (y == 0) | (y == 1)
        X_raw, y = X_raw[mask], y[mask]

    if (y == 1).sum() == 0:
        print("  Sin muestras preictales")
        return None

    X_raw, y = remove_emg_artifacts(X_raw, y)

    n_pre = int((y == 1).sum())
    n_inter = int((y == 0).sum())
    print(
        f"  Dataset: {X_raw.shape[0]} ventanas "
        f"(pre={n_pre}, inter={n_inter}, ratio={n_inter / max(n_pre, 1):.1f}:1)"
    )

    # STFT espectrogramas
    print("  Generando espectrogramas STFT...")
    X_spec = compute_spectrogram_v4(X_raw)
    print(f"  Espectrogramas: {X_spec.shape}")

    # Features SVM
    print("  Extrayendo features para SVM...")
    X_svm = extract_svm_features_v4(X_spec)
    print(f"  Features SVM: {X_svm.shape}")

    del X_raw
    _gc.collect()

    return {"X_spec": X_spec, "X_svm": X_svm, "y": y}
