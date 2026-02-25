#!/usr/bin/env python3
"""
Pipeline Fase 1 — Partición aleatoria + LOSO SVM
=================================================

Reproduce los resultados de las Tablas 4.1–4.3 del TFM (Fase 1):
  - Tabla 4.1: Random-partition SeizureLSTM con evaluación event-based.
  - Tabla 4.2: Random-partition SeizureTransformer con evaluación event-based.
  - Tabla 4.3: LOSO SVM + ImprovedCNNTransformer (Flujo.ipynb).

También evalúa segment-based MTPW para los modelos de random partition.

Uso::

    python run_phase1_pipeline.py --raw_dir  /ruta/a/CHBMIT
                                  --out_dir  /ruta/a/outputs
                                  [--subjects chb01 chb02 ...]

Requiere:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Importaciones del paquete ──────────────────────────────────
from seizure_prediction.config import (
    ALARM_THRESHOLD,
    ALARM_WINDOW_MIN,
    DEVICE,
    FS,
    MTPW_LENGTHS,
    SEED,
    SOP_MIN,
    SPH_MIN,
    STFT_HOP,
    STFT_NFFT,
    WIN_SEC,
)
from seizure_prediction.data import (
    compute_absolute_seizure_times,
    get_valid_subjects,
    load_subject_continuous,
    parse_summary,
    select_leading_seizures,
)
from seizure_prediction.evaluation import (
    event_based_evaluation,
    predict_temporal,
    predict_with_spectrogram,
    segment_based_mtpw,
)
from seizure_prediction.preprocessing import process_subject
from seizure_prediction.training import (
    train_subject_model,
    train_subject_svm,
    train_subject_transformer,
)


# ================================================================
# Random-partition pipeline (SeizureLSTM + SeizureTransformer)
# ================================================================

def run_random_partition(
    raw_dir: Path,
    out_dir: Path,
    subjects: list[str],
) -> dict:
    """Ejecuta pipeline Fase 1 con partición aleatoria 10-fold CV.

    Entrena SeizureLSTM y SeizureTransformer por paciente y evalúa
    con métricas segment-based (MTPW) y event-based (alarma v1).
    """
    np.random.seed(SEED)
    results: dict = {"lstm": {}, "transformer": {}}

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  Procesando {subj} — Random partition")
        print(f"{'='*60}")

        # 1. Cargar datos
        subj_dir = raw_dir / subj
        summary_path = subj_dir / f"{subj}-summary.txt"
        if not summary_path.exists():
            print(f"  ⚠ No se encontró summary para {subj}, saltando.")
            continue

        records = parse_summary(summary_path)
        seizures_abs = compute_absolute_seizure_times(records)
        leading = select_leading_seizures(seizures_abs)
        if len(leading) < 3:
            print(f"  ⚠ {subj}: solo {len(leading)} leading seizures, saltando.")
            continue

        # 2. Preprocesar (gamma + normalización + undersampling)
        result = process_subject(subj, raw_dir, seizures_abs)
        if result is None:
            print(f"  ⚠ {subj}: error en preprocesamiento, saltando.")
            continue
        X, y = result

        # 3. Entrenar SeizureLSTM
        print(f"  Entrenando SeizureLSTM... ({X.shape[0]} ventanas)")
        lstm_result = train_subject_model(X, y, n_splits=10)
        results["lstm"][subj] = lstm_result
        print(f"  → LSTM Sen={lstm_result['sensitivity']:.3f}  "
              f"Spec={lstm_result['specificity']:.3f}  "
              f"F1={lstm_result['f1']:.3f}")

        # 4. Entrenar SeizureTransformer
        print(f"  Entrenando SeizureTransformer...")
        tf_result = train_subject_transformer(X, y, n_splits=10)
        results["transformer"][subj] = tf_result
        print(f"  → Transformer Sen={tf_result['sensitivity']:.3f}  "
              f"Spec={tf_result['specificity']:.3f}  "
              f"F1={tf_result['f1']:.3f}")

        # 5. Evaluar event-based v1 (último fold como ejemplo)
        total_preds = lstm_result.get("all_preds")
        total_labels = lstm_result.get("all_labels")
        if total_preds is not None and total_labels is not None:
            ev = event_based_evaluation(
                total_preds, total_labels,
                alarm_window_min=ALARM_WINDOW_MIN,
                alarm_threshold=ALARM_THRESHOLD,
                sph_min=SPH_MIN,
                sop_min=SOP_MIN,
            )
            results["lstm"][subj]["event_based"] = ev
            print(f"  → Event-based LSTM: Sen={ev['sensitivity']:.3f}  "
                  f"FAR={ev['far_per_hour']:.3f}/h")

        # 6. MTPW
        if total_preds is not None:
            mtpw = {}
            for L in MTPW_LENGTHS:
                mtpw[L] = segment_based_mtpw(total_preds, total_labels, L)
            results["lstm"][subj]["mtpw"] = mtpw

        # 7. SVM (Fase 1: gamma features, PCA 95%)
        print(f"  Entrenando SVM (gamma)...")
        svm_result = train_subject_svm(X, y, n_splits=10)
        results.setdefault("svm", {})[subj] = svm_result
        print(f"  → SVM Sen={svm_result['sensitivity']:.3f}  "
              f"Spec={svm_result['specificity']:.3f}  "
              f"F1={svm_result['f1']:.3f}")

    return results


# ================================================================
# Resumen de resultados
# ================================================================

def print_summary(results: dict) -> None:
    """Imprime promedios (macro) por modelo."""
    for model_name in results:
        model_results = results[model_name]
        if not model_results:
            continue

        sens = [r["sensitivity"] for r in model_results.values()]
        specs = [r["specificity"] for r in model_results.values()]
        f1s = [r["f1"] for r in model_results.values()]
        accs = [r["accuracy"] for r in model_results.values()]

        print(f"\n{'='*40}")
        print(f"  {model_name.upper()} — Promedio sobre {len(sens)} pacientes")
        print(f"{'='*40}")
        print(f"  Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        print(f"  Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
        print(f"  F1-score:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Fase 1 — Random partition + LOSO SVM",
    )
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="Directorio raíz de CHB-MIT (con subcarpetas chbXX).")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_phase1"),
                        help="Directorio de salida para resultados.")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Lista de sujetos (e.g. chb01 chb02)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolver sujetos
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_valid_subjects(args.raw_dir)
    print(f"Sujetos a procesar: {subjects}")

    # Ejecutar pipeline principal
    results = run_random_partition(args.raw_dir, args.out_dir, subjects)

    # Resumen
    print_summary(results)

    # Guardar JSON
    out_file = args.out_dir / "phase1_results.json"

    def _convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"\nResultados guardados en {out_file}")


if __name__ == "__main__":
    main()
