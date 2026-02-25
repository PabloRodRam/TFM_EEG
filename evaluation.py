"""
Evaluación segment-based y event-based de predicción de crisis.

Incluye:
  - **Segment-based**: predicción temporal, MTPW (majority voting).
  - **Event-based v1**: alarma con ventana deslizante (SPH=30 min, SOP=20 min).
  - **Event-based v4**: firing rule *k-of-n* + SPH=5 min, SOP=30 min.
  - Predicción con espectrogramas (transformer) y SVM.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    DEVICE,
    FIRING_K,
    FIRING_N,
    FS,
    STFT_HOP,
    STFT_NFFT,
    WIN_SEC,
)
from .features import extract_features_batch


# ================================================================
# Predicción temporal (sin shuffle)
# ================================================================

@torch.no_grad()
def predict_temporal(
    model: nn.Module,
    X_temporal: np.ndarray,
    batch_size: int = 256,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """Predice sobre datos temporales en orden (sin shuffle)."""
    model.eval()
    X_t = torch.tensor(X_temporal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    all_preds: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


@torch.no_grad()
def predict_with_spectrogram(
    model: nn.Module,
    X_raw: np.ndarray,
    n_fft: int = STFT_NFFT,
    hop_length: int = STFT_HOP,
    batch_size: int = 256,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """Calcula espectrograma on-the-fly + predice en chunks (ahorra RAM)."""
    model.eval()
    n = len(X_raw)
    win_samples = X_raw.shape[1]
    n_ch = X_raw.shape[2]
    all_preds: list[np.ndarray] = []

    window = torch.hann_window(n_fft).to(device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(
            X_raw[start:end], dtype=torch.float32, device=device,
        )
        bs = end - start

        flat = batch.permute(0, 2, 1).reshape(-1, win_samples)
        Zxx = torch.stft(
            flat, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, window=window, center=True, return_complex=True,
        )
        mag = torch.log1p(torch.abs(Zxx))
        nf, nt = mag.shape[1], mag.shape[2]
        X_spec_batch = mag.reshape(bs, n_ch, nf, nt)

        logits = model(X_spec_batch)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())

        del batch, flat, Zxx, mag, X_spec_batch, logits

    torch.cuda.empty_cache()
    return np.concatenate(all_preds)


def predict_svm_temporal(
    svm_model,
    scaler,
    pca,
    X_raw: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """Extrae features + normaliza + PCA + predice con SVM."""
    X_feat = extract_features_batch(X_raw, fs=fs)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_feat = scaler.transform(X_feat)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    if pca is not None:
        X_feat = pca.transform(X_feat)
    return svm_model.predict(X_feat)


# ================================================================
# MTPW — Majority Voting (segment-based)
# ================================================================

def segment_based_mtpw(
    preds_5s: np.ndarray,
    labels_5s: np.ndarray,
    L_seconds: int,
    win_sec: int = WIN_SEC,
) -> dict:
    """Majority voting (MTPW): agrupa L/5 ventanas y aplica voto mayoritario.

    Longitudes evaluadas: 5, 30, 60, 90, 120, 150 s.
    """
    n_per_seg = max(1, L_seconds // win_sec)
    n_segments = len(preds_5s) // n_per_seg

    if n_segments == 0:
        return {
            "sensitivity": 0, "specificity": 0,
            "accuracy": 0, "f1": 0, "n_segments": 0,
        }

    seg_preds = np.zeros(n_segments, dtype=int)
    seg_labels = np.zeros(n_segments, dtype=int)

    for i in range(n_segments):
        start = i * n_per_seg
        end = start + n_per_seg
        seg_preds[i] = 1 if (preds_5s[start:end] == 1).sum() > n_per_seg / 2 else 0
        seg_labels[i] = 1 if (labels_5s[start:end] == 1).sum() > n_per_seg / 2 else 0

    sen = recall_score(seg_labels, seg_preds, pos_label=1, zero_division=0)
    spec = recall_score(seg_labels, seg_preds, pos_label=0, zero_division=0)
    acc = accuracy_score(seg_labels, seg_preds)
    f1 = f1_score(seg_labels, seg_preds, pos_label=1, zero_division=0)

    return {
        "sensitivity": sen,
        "specificity": spec,
        "accuracy": acc,
        "f1": f1,
        "n_segments": n_segments,
    }


# ================================================================
# Event-based evaluation v1
# ================================================================

def event_based_evaluation(
    preds_5s: np.ndarray,
    labels_5s: np.ndarray,
    win_sec: int = WIN_SEC,
    alarm_window_min: int = 10,
    alarm_threshold: float = 0.70,
    sph_min: int = 30,
    sop_min: int = 20,
    n_skip_seizures: int = 0,
) -> dict:
    """Evaluación event-based v1 con ventana deslizante y SPH/SOP.

    1. Ventana deslizante de *alarm_window_min* → alarma si > threshold.
    2. TP: crisis TEST entre ``t_alarm + SPH`` y ``t_alarm + SPH + SOP``.
    3. FAR = false_alarms / horas_interictal.
    4. Warning time = crisis_onset − t_alarm.
    """
    n_windows = len(preds_5s)
    alarm_win_size = (alarm_window_min * 60) // win_sec
    sph_size = (sph_min * 60) // win_sec
    sop_size = (sop_min * 60) // win_sec

    # Detectar alarmas
    alarms: list[int] = []
    i = 0
    while i + alarm_win_size <= n_windows:
        frac = (preds_5s[i: i + alarm_win_size] == 1).mean()
        if frac > alarm_threshold:
            alarms.append(i)
            i += alarm_win_size
        else:
            i += 1

    # Detectar TODAS las crisis (transición preictal → no-preictal)
    all_seizure_onsets: list[int] = []
    in_preictal = False
    for j in range(len(labels_5s)):
        if labels_5s[j] == 1 and not in_preictal:
            in_preictal = True
        elif labels_5s[j] != 1 and in_preictal:
            in_preictal = False
            all_seizure_onsets.append(j)
    if in_preictal:
        all_seizure_onsets.append(len(labels_5s))

    train_onsets = all_seizure_onsets[:n_skip_seizures]
    test_onsets = all_seizure_onsets[n_skip_seizures:]

    n_seizures = len(test_onsets)
    interictal_hours = (labels_5s == 0).sum() * win_sec / 3600.0

    seizure_predicted = [False] * n_seizures
    warning_times: list[float] = []
    false_alarms = 0

    for alarm_idx in alarms:
        # ¿predice crisis de entrenamiento?
        matches_train = False
        for train_onset in train_onsets:
            ps = alarm_idx + sph_size
            pe = alarm_idx + sph_size + sop_size
            if ps <= train_onset <= pe:
                matches_train = True
                break
        if matches_train:
            continue

        # Matching con crisis de test
        matched = False
        for sz_i, sz_onset in enumerate(test_onsets):
            ps = alarm_idx + sph_size
            pe = alarm_idx + sph_size + sop_size
            if ps <= sz_onset <= pe:
                if not seizure_predicted[sz_i]:
                    seizure_predicted[sz_i] = True
                    wt = (sz_onset - alarm_idx) * win_sec / 60.0
                    warning_times.append(wt)
                    matched = True
                    break
        if not matched:
            false_alarms += 1

    sensitivity = sum(seizure_predicted) / max(n_seizures, 1)
    far_per_hour = false_alarms / max(interictal_hours, 0.01)
    mean_warning = float(np.mean(warning_times)) if warning_times else 0.0

    return {
        "n_seizures": n_seizures,
        "n_predicted": sum(seizure_predicted),
        "sensitivity": sensitivity,
        "false_alarms": false_alarms,
        "far_per_hour": far_per_hour,
        "interictal_hours": interictal_hours,
        "mean_warning_time_min": mean_warning,
        "n_alarms": len(alarms),
        "warning_times": warning_times,
    }


# ================================================================
# Firing rule (postprocesado v4)
# ================================================================

def apply_firing_rule(
    preds_5s: np.ndarray,
    k: int = FIRING_K,
    n: int = FIRING_N,
) -> np.ndarray:
    """Aplica firing rule *k-of-n* a predicciones de ventanas de 5 s.

    Una ventana se considera «alarma activa» si al menos *k* de las
    últimas *n* ventanas (incluida la actual) son positivas.
    """
    result = np.zeros_like(preds_5s)
    cumsum = np.cumsum(np.concatenate([[0], preds_5s]))

    for i in range(len(preds_5s)):
        start = max(0, i - n + 1)
        window_sum = cumsum[i + 1] - cumsum[start]
        if window_sum >= k:
            result[i] = 1

    return result


# ================================================================
# Event-based evaluation v4 (firing rule)
# ================================================================

def event_based_evaluation_v4(
    preds_5s: np.ndarray,
    labels_5s: np.ndarray,
    win_sec: int = WIN_SEC,
    firing_k: int = FIRING_K,
    firing_n: int = FIRING_N,
    sph_min: int = 5,
    sop_min: int = 30,
    n_skip_seizures: int = 0,
    seizure_onsets: list[int] | None = None,
) -> dict:
    """Evaluación event-based v4 con firing rule.

    Mejoras respecto a v1:
      - Firing rule *k-of-n* antes de detectar alarmas.
      - SPH = 5 min, SOP = 30 min.
      - Alarma = primera ventana con ``fired = 1`` tras un periodo sin alarma.
      - Soporte para *seizure_onsets* explícitos.
    """
    fired = apply_firing_rule(preds_5s, k=firing_k, n=firing_n)

    sph_size = (sph_min * 60) // win_sec
    sop_size = (sop_min * 60) // win_sec

    # Detectar alarmas (transiciones 0 → 1)
    alarms: list[int] = []
    in_alarm = False
    for i in range(len(fired)):
        if fired[i] == 1 and not in_alarm:
            alarms.append(i)
            in_alarm = True
        elif fired[i] == 0:
            in_alarm = False

    # Detectar crisis
    if seizure_onsets is not None:
        all_seizure_onsets = list(seizure_onsets)
    else:
        all_seizure_onsets = []
        in_preictal = False
        for j in range(len(labels_5s)):
            if labels_5s[j] == 1 and not in_preictal:
                in_preictal = True
            elif labels_5s[j] != 1 and in_preictal:
                in_preictal = False
                all_seizure_onsets.append(j)
        if in_preictal:
            all_seizure_onsets.append(len(labels_5s))

    train_onsets = all_seizure_onsets[:n_skip_seizures]
    test_onsets = all_seizure_onsets[n_skip_seizures:]

    n_seizures = len(test_onsets)
    interictal_hours = (labels_5s == 0).sum() * win_sec / 3600.0

    seizure_predicted = [False] * n_seizures
    warning_times: list[float] = []
    false_alarms = 0

    for alarm_idx in alarms:
        matches_train = False
        for train_onset in train_onsets:
            ps = alarm_idx + sph_size
            pe = alarm_idx + sph_size + sop_size
            if ps <= train_onset <= pe:
                matches_train = True
                break
        if matches_train:
            continue

        matched = False
        for sz_i, sz_onset in enumerate(test_onsets):
            ps = alarm_idx + sph_size
            pe = alarm_idx + sph_size + sop_size
            if ps <= sz_onset <= pe:
                if not seizure_predicted[sz_i]:
                    seizure_predicted[sz_i] = True
                    wt = (sz_onset - alarm_idx) * win_sec / 60.0
                    warning_times.append(wt)
                    matched = True
                    break
        if not matched:
            false_alarms += 1

    sensitivity = sum(seizure_predicted) / max(n_seizures, 1)
    far_per_hour = false_alarms / max(interictal_hours, 0.01)
    mean_warning = float(np.mean(warning_times)) if warning_times else 0.0

    return {
        "n_seizures": n_seizures,
        "n_predicted": sum(seizure_predicted),
        "sensitivity": sensitivity,
        "false_alarms": false_alarms,
        "far_per_hour": far_per_hour,
        "interictal_hours": interictal_hours,
        "mean_warning_time_min": mean_warning,
        "n_alarms": len(alarms),
        "warning_times": warning_times,
    }
