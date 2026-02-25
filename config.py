"""
Constantes globales y configuración del pipeline.

Centraliza todos los hiperparámetros para las tres fases del TFM:
  - Fase 1 (v1): banda gamma 30-128 Hz, partición aleatoria.
  - Fase 3 (v4): banda broadband 0.5-128 Hz, partición cronológica.
"""

from pathlib import Path

import torch

# ============================================================
# Rutas del proyecto
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "CHBMIT"
OUT_DIR = ROOT / "data" / "processed"

# ============================================================
# Señal EEG
# ============================================================
FS = 256                     # Frecuencia de muestreo CHB-MIT (Hz)
WIN_SEC = 5                  # Ventana básica (s)
WIN_SAMPLES = FS * WIN_SEC   # 1 280 muestras por ventana
N_CHANNELS = 23              # Canales bipolares estándar 10-20

# ============================================================
# Filtrado — Fase 1 (banda gamma)
# ============================================================
GAMMA_LO = 30.0              # Hz
GAMMA_HI = 128.0             # Hz (Nyquist = 128 Hz)
BUTTER_ORDER = 4

# ============================================================
# Filtrado — Fase 3 (broadband)
# ============================================================
BROADBAND_LO = 0.5           # Hz
BROADBAND_HI = 128.0         # Hz

# ============================================================
# Etiquetado temporal
# ============================================================
PREICTAL_MIN = 60            # Ventana preictal antes de cada crisis (min)
POSTICTAL_MIN = 60           # Ventana postictal después de cada crisis (min)
INTER_GAP_POST = 5 * 3600   # Gap interictal post-crisis (s) — 5 h
INTER_GAP_PRE = 3 * 3600    # Gap interictal pre-crisis (s) — 3 h
LEADING_GAP_MIN = 60         # Gap mínimo entre leading seizures (min)
MIN_LEADING = 3              # Mínimo de leading seizures por sujeto

# ============================================================
# Canales estándar bipolares 10-20 (23 canales)
# ============================================================
STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",              # Temporal izquierdo
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",              # Parasagital izquierdo
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",              # Parasagital derecho
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",              # Temporal derecho
    "FZ-CZ", "CZ-PZ",                                  # Línea media
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8",  # Suplementarios
]

# ============================================================
# Espectrogramas STFT
# ============================================================
# Fase 1 (gamma)
STFT_NFFT = 256
STFT_HOP = 32

# Fase 3 (broadband, v4)
STFT_NFFT_V4 = 256
STFT_HOP_V4 = 128

# ============================================================
# Bandas de frecuencia para features SVM v4
# ============================================================
FREQ_BANDS = {
    "delta":  (0.5, 4),
    "theta":  (4, 8),
    "alpha":  (8, 13),
    "beta":   (13, 30),
    "gamma":  (30, 128),
}

# ============================================================
# Firing rule (postprocesado temporal, Fase 3)
# ============================================================
FIRING_N = 20                # Tamaño de ventana de firing
FIRING_K = 12                # Mínimo de positivos para activar

# ============================================================
# Entrenamiento
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ============================================================
# Evaluación event-based — Fase 1
# ============================================================
ALARM_WINDOW_MIN = 10        # Ventana de alarma (min)
ALARM_THRESHOLD = 0.70       # Proporción para confirmar alarma
SPH_MIN = 30                 # Seizure Prediction Horizon (min)
SOP_MIN = 20                 # Seizure Occurrence Period (min)

# ============================================================
# Evaluación event-based — Fase 3 (v4)
# ============================================================
SPH_MIN_V4 = 5               # SPH para v4 (min)
SOP_MIN_V4 = 30              # SOP para v4 (min)

# ============================================================
# MTPW (Medium-Term Prediction Window)
# ============================================================
MTPW_LENGTHS = [5, 30, 60, 90, 120, 150]
