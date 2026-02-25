#!/usr/bin/env python3
"""
Pipeline Fase 3 — Protocolo corregido (v4)
==========================================

Reproduce los resultados de las Tablas 5.1–5.9 del TFM (Fase 3):
  - Tabla 5.1-5.3: Segment-based CNN-LSTM, SVM, Transformer (broadband).
  - Tabla 5.4-5.6: Event-based con firing rule k-of-n.
  - Tabla 5.7-5.9: MTPW para los tres modelos corregidos.

Cambios clave respecto a Fase 1:
  - Filtro broadband (0.5–128 Hz) en lugar de gamma.
  - Partición cronológica por crisis (2 train + resto test).
  - Sin undersampling.
  - STFT con hop=128 (espectrogramas menos granulares).
  - Firing rule *k-of-n* (k=12, n=20) como postprocesado.
  - SPH = 5 min, SOP = 30 min.

Uso::

    python run_phase3_pipeline.py --raw_dir  /ruta/a/CHBMIT
                                  --out_dir  /ruta/a/outputs_v4
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
    DEVICE,
    FIRING_K,
    FIRING_N,
    FS,
    MTPW_LENGTHS,
    SEED,
    SOP_MIN_V4,
    SPH_MIN_V4,
    STFT_HOP_V4,
    STFT_NFFT_V4,
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
    apply_firing_rule,
    event_based_evaluation_v4,
    predict_temporal,
    predict_with_spectrogram,
    predict_svm_temporal,
    segment_based_mtpw,
)
from seizure_prediction.preprocessing import process_subject_v4
from seizure_prediction.training import (
    calibrate_firing_k,
    split_train_test_by_seizure,
    train_cnnlstm_v4,
    train_svm_v4,
    train_transformer_v4,
)


# ================================================================
# Pipeline principal Fase 3 (v4)
# ================================================================

def run_corrected_pipeline(
    raw_dir: Path,
    out_dir: Path,
    subjects: list[str],
) -> dict:
    """Ejecuta pipeline Fase 3: partición cronológica + firing rule.

    Para cada paciente:
      1. Preprocesa con broadband + segmenta + genera espectrogramas v4.
      2. Split cronológico 2 seizures train / resto test.
      3. Entrena CNN-LSTM, SVM y Transformer (10-fold AUC en train).
      4. Evalúa segment-based, event-based (firing rule), MTPW.
    """
    np.random.seed(SEED)
    results: dict = {"cnnlstm": {}, "svm": {}, "transformer": {}}

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  Procesando {subj} — Protocolo corregido (v4)")
        print(f"{'='*60}")

        # 1. Cargar y parsear
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

        # 2. Preprocesar broadband (sin undersampling)
        result = process_subject_v4(subj, raw_dir, seizures_abs)
        if result is None:
            print(f"  ⚠ {subj}: error en preprocesamiento v4, saltando.")
            continue
        X_raw, X_spec, X_svm, y = result

        # 3. Split cronológico
        X_train, y_train, X_test, y_test = split_train_test_by_seizure(
            subj, seizures_abs, X_raw, y, n_train_seizures=2,
        )

        # Sub-splits para spectrograms y SVM
        X_spec_train, _, X_spec_test, _ = split_train_test_by_seizure(
            subj, seizures_abs, X_spec, y, n_train_seizures=2,
        )
        X_svm_train, _, X_svm_test, _ = split_train_test_by_seizure(
            subj, seizures_abs, X_svm, y, n_train_seizures=2,
        )

        n_train = len(y_train)
        n_test = len(y_test)
        print(f"  Train: {n_train} ({(y_train==1).sum()} preictal)  "
              f"Test: {n_test} ({(y_test==1).sum()} preictal)")

        # ── CNN-LSTM ──────────────────────────────────────
        print(f"  Entrenando CNN-LSTM v4...")
        cnnlstm_result = train_cnnlstm_v4(
            X_train, y_train, X_test, y_test,
        )
        results["cnnlstm"][subj] = cnnlstm_result
        print(f"  → CNN-LSTM  Sen={cnnlstm_result['sensitivity']:.3f}  "
              f"Spec={cnnlstm_result['specificity']:.3f}  "
              f"F1={cnnlstm_result['f1']:.3f}")

        # Evaluación event-based sobre test
        preds_cnnlstm = predict_temporal(
            cnnlstm_result["model"], X_test, device=DEVICE,
        )
        ev_cnnlstm = event_based_evaluation_v4(
            preds_cnnlstm, y_test,
            firing_k=FIRING_K, firing_n=FIRING_N,
            sph_min=SPH_MIN_V4, sop_min=SOP_MIN_V4,
        )
        results["cnnlstm"][subj]["event_based"] = ev_cnnlstm

        # MTPW
        mtpw_cnnlstm = {}
        for L in MTPW_LENGTHS:
            mtpw_cnnlstm[L] = segment_based_mtpw(preds_cnnlstm, y_test, L)
        results["cnnlstm"][subj]["mtpw"] = mtpw_cnnlstm

        # ── SVM v4 ────────────────────────────────────────
        print(f"  Entrenando SVM v4...")
        svm_result = train_svm_v4(
            X_svm_train, y_train, X_svm_test, y_test,
        )
        results["svm"][subj] = svm_result
        print(f"  → SVM  Sen={svm_result['sensitivity']:.3f}  "
              f"Spec={svm_result['specificity']:.3f}  "
              f"F1={svm_result['f1']:.3f}")

        preds_svm = svm_result.get("preds_test")
        if preds_svm is not None:
            ev_svm = event_based_evaluation_v4(
                preds_svm, y_test,
                firing_k=FIRING_K, firing_n=FIRING_N,
                sph_min=SPH_MIN_V4, sop_min=SOP_MIN_V4,
            )
            results["svm"][subj]["event_based"] = ev_svm

            mtpw_svm = {}
            for L in MTPW_LENGTHS:
                mtpw_svm[L] = segment_based_mtpw(preds_svm, y_test, L)
            results["svm"][subj]["mtpw"] = mtpw_svm

        # ── Transformer v4 ───────────────────────────────
        print(f"  Entrenando Transformer v4...")
        tf_result = train_transformer_v4(
            X_spec_train, y_train, X_spec_test, y_test,
        )
        results["transformer"][subj] = tf_result
        print(f"  → Transformer  Sen={tf_result['sensitivity']:.3f}  "
              f"Spec={tf_result['specificity']:.3f}  "
              f"F1={tf_result['f1']:.3f}")

        preds_tf = tf_result.get("preds_test")
        if preds_tf is not None:
            ev_tf = event_based_evaluation_v4(
                preds_tf, y_test,
                firing_k=FIRING_K, firing_n=FIRING_N,
                sph_min=SPH_MIN_V4, sop_min=SOP_MIN_V4,
            )
            results["transformer"][subj]["event_based"] = ev_tf

            mtpw_tf = {}
            for L in MTPW_LENGTHS:
                mtpw_tf[L] = segment_based_mtpw(preds_tf, y_test, L)
            results["transformer"][subj]["mtpw"] = mtpw_tf

    return results


# ================================================================
# Resumen
# ================================================================

def print_summary(results: dict) -> None:
    """Promedios (macro) por modelo — segment-based y event-based."""
    for model_name in results:
        model_results = results[model_name]
        if not model_results:
            continue

        sens = [r["sensitivity"] for r in model_results.values() if "sensitivity" in r]
        specs = [r["specificity"] for r in model_results.values() if "specificity" in r]
        f1s = [r["f1"] for r in model_results.values() if "f1" in r]
        accs = [r["accuracy"] for r in model_results.values() if "accuracy" in r]

        print(f"\n{'='*50}")
        print(f"  {model_name.upper()} — Promedio {len(sens)} pacientes")
        print(f"{'='*50}")
        if sens:
            print(f"  Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        if specs:
            print(f"  Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
        if f1s:
            print(f"  F1-score:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        if accs:
            print(f"  Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")

        # Event-based promedio
        ev_sens = [
            r["event_based"]["sensitivity"]
            for r in model_results.values()
            if "event_based" in r
        ]
        ev_fars = [
            r["event_based"]["far_per_hour"]
            for r in model_results.values()
            if "event_based" in r
        ]
        if ev_sens:
            print(f"\n  Event-based:")
            print(f"    Sensitivity:  {np.mean(ev_sens):.4f} ± {np.std(ev_sens):.4f}")
            print(f"    FAR/h:        {np.mean(ev_fars):.4f} ± {np.std(ev_fars):.4f}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Fase 3 — Protocolo corregido (v4)",
    )
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="Directorio raíz de CHB-MIT (con subcarpetas chbXX).")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_phase3"),
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

    # Ejecutar pipeline v4
    results = run_corrected_pipeline(args.raw_dir, args.out_dir, subjects)

    # Resumen
    print_summary(results)

    # Guardar JSON
    out_file = args.out_dir / "phase3_results.json"

    def _convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if hasattr(o, "state_dict"):
            return "<torch.nn.Module>"
        return str(o)

    # Eliminar modelos del JSON (no serializables)
    results_clean = {}
    for model_name, subj_dict in results.items():
        results_clean[model_name] = {}
        for subj, metrics in subj_dict.items():
            results_clean[model_name][subj] = {
                k: v for k, v in metrics.items() if k != "model"
            }

    with open(out_file, "w") as f:
        json.dump(results_clean, f, indent=2, default=_convert)
    print(f"\nResultados guardados en {out_file}")


if __name__ == "__main__":
    main()
