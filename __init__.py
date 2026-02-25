"""
seizure_prediction — Paquete para predicción de crisis epilépticas.

TFM: Comparativa de SVM, LSTM y CNN-Transformer para la predicción
de crisis epilépticas a partir de EEG de superficie (CHB-MIT).
Incluye diagnóstico y corrección de data leakage.

Fases experimentales:
    Fase 1 — Implementación inicial (partición aleatoria, banda gamma).
    Fase 2 — Diagnóstico del data leakage.
    Fase 3 — Protocolo corregido (partición cronológica, broadband).
"""

__version__ = "1.0.0"
