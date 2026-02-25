# Seizure Prediction — EEG-based (CHB-MIT)

**TFM: Comparativa de modelos de aprendizaje automático para la predicción de crisis epilépticas: diagnóstico y cuantificación del data leakage en protocolos de evaluación con EEG.**

> Máster Universitario en Inteligencia Artificial — UNIR, 2025/2026.

---

## Resumen

Este repositorio contiene el código fuente del Trabajo Fin de Máster (TFM) sobre **predicción de crisis epilépticas** a partir de señales EEG de superficie, utilizando la base de datos **CHB-MIT Scalp EEG** (24 pacientes pediátricos, 982 horas, 198 crisis, 256 Hz).

Se implementan y comparan tres familias de modelos:
- **SVM** (Support Vector Machine) con features estadísticas y espectrales.
- **LSTM / BiLSTM / CNN-LSTM** (redes recurrentes).
- **CNN-Transformer** (atención multi-cabeza sobre espectrogramas STFT).

El trabajo incluye un **diagnóstico y corrección de data leakage** que demuestra la importancia de la partición cronológica en datos EEG.

---

## Estructura del código

```
├── requirements.txt              # Dependencias
├── .gitignore                    # Archivos excluidos
├── run_phase1_pipeline.py        # Pipeline Fase 1 (random partition)
├── run_phase3_pipeline.py        # Pipeline Fase 3 (protocolo corregido)
├── README.md                     # Este archivo
├── __init__.py               # Metadatos del paquete
├── config.py                 # Constantes e hiperparámetros
├── data.py                   # Carga y parsing de CHB-MIT
├── preprocessing.py          # Filtrado, segmentación, etiquetado
├── features.py               # Extracción de features y STFT
├── models.py                 # Arquitecturas DL (5 modelos + FocalLoss)
├── training.py               # Entrenamiento, splits, calibración
└── evaluation.py             # Evaluación segment/event-based, firing rule
```

---

## Fases experimentales

### Fase 1 — Implementación inicial (Tablas 4.1–4.3)

| Aspecto | Configuración |
|---------|---------------|
| Filtro | Gamma 30–128 Hz (Butterworth orden 4) |
| Partición | **Aleatoria** (10-fold CV estratificado) |
| Ventana | 5 s (1 280 muestras × 23 canales) |
| Features SVM | 12 por canal × 23 = **276** |
| Espectrograma | STFT n_fft=256, hop=32 |
| Evaluación event-based | Alarma ventana deslizante 10 min @70%, SPH=30 min, SOP=20 min |
| Modelos | SeizureLSTM, SeizureTransformer, SVM |

**Resultado**: Sensibilidades >90% que se diagnostican como **infladas** por data leakage en Fase 2.

### Fase 2 — Diagnóstico de data leakage

Se identifica que la partición aleatoria mezcla ventanas temporalmente adyacentes entre train y test, provocando fugas de información. Se demuestra con:
- Análisis de autocorrelación de ventanas pre/post-ictales.
- Comparación respuesta modelo con ventanas consecutivas en train vs test.

### Fase 3 — Protocolo corregido (Tablas 5.1–5.9)

| Aspecto | Configuración |
|---------|---------------|
| Filtro | Broadband 0.5–128 Hz (Butterworth orden 4) |
| Partición | **Cronológica** (2 primeras crisis → train, resto → test) |
| Ventana | 5 s (1 280 muestras × 23 canales) |
| Features SVM v4 | 5 bandas × 3 stats × 23 ch = **345** |
| Espectrograma v4 | STFT n_fft=256, hop=128 |
| Firing rule | k-of-n (k=12, n=20) |
| Evaluación event-based | SPH=5 min, SOP=30 min |
| Modelos | SeizureCNNLSTM, SeizureTransformer, SVM |

**Resultado**: Caída significativa de rendimiento respecto a Fase 1, confirmando el efecto del data leakage.

---

## Modelos implementados

| Modelo | Entrada | Salida | Loss | Archivo |
|--------|---------|--------|------|---------|
| `SeizureLSTM` | Temporal (N, T, C) | 2 clases | CrossEntropy | `models.py` |
| `SeizureBiLSTM` | Temporal (N, T, C) | 2 clases | CrossEntropy | `models.py` |
| `SeizureCNNLSTM` | Temporal (N, T, C) | 2 clases | CrossEntropy | `models.py` |
| `SeizureTransformer` | Espectrograma (N, C, F, T) | 2 clases | CrossEntropy | `models.py` |
| `ImprovedCNNTransformer` | Espectrograma (N, C, F, T) | 1 logit | FocalLoss (BCE) | `models.py` |
| `SVM (SVC)` | Features vector | Binario | Hinge (SVC) | `training.py` |

---

## Instalación

```bash
# Clonar
git clone https://github.com/<usuario>/eeg-seizure-prediction.git
cd eeg-seizure-prediction

# Entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Dependencias
pip install -r requirements.txt
```

### Requisitos
- Python ≥ 3.10
- PyTorch ≥ 2.1 (con CUDA para GPU)
- MNE-Python ≥ 1.6
- scikit-learn ≥ 1.3
- NumPy, SciPy, Pandas

---

## Uso

### Fase 1 (random partition)

```bash
python run_phase1_pipeline.py \
    --raw_dir /ruta/a/CHBMIT \
    --out_dir outputs_phase1 \
    --subjects chb01 chb02 chb03
```

### Fase 3 (protocolo corregido)

```bash
python run_phase3_pipeline.py \
    --raw_dir /ruta/a/CHBMIT \
    --out_dir outputs_phase3 \
    --subjects chb01 chb02 chb03
```

### Uso como librería

```python
from seizure_prediction.data import parse_summary, load_subject_continuous
from seizure_prediction.preprocessing import process_subject_v4
from seizure_prediction.models import SeizureCNNLSTM
from seizure_prediction.training import train_cnnlstm_v4
from seizure_prediction.evaluation import event_based_evaluation_v4

# Cargar y preprocesar
records = parse_summary("chb01/chb01-summary.txt")
result = process_subject_v4("chb01", raw_dir, seizures_abs)
X_raw, X_spec, X_svm, y = result

# Entrenar
metrics = train_cnnlstm_v4(X_train, y_train, X_test, y_test)

# Evaluar
ev = event_based_evaluation_v4(preds, labels, firing_k=12, firing_n=20)
print(f"Sensitivity: {ev['sensitivity']:.3f}, FAR/h: {ev['far_per_hour']:.3f}")
```

---

## Datos

Este proyecto utiliza la base de datos **CHB-MIT Scalp EEG Database**:

> Shoeb, A. H. (2009). Application of Machine Learning to Epileptic Seizure
> Onset Detection and Treatment. MIT PhD thesis.

Disponible en [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

**No se incluyen datos en el repositorio**. Descargar los archivos `.edf` y colocarlos en `data/raw/CHBMIT/chbXX/`.

---

## Referencia

Si utilizas este código, por favor cita:

```
@mastersthesis{tfm_eeg_2025,
  author = {Pablo},
  title  = {Comparativa de SVM, LSTM y CNN-Transformer para la predicción
            de crisis epilépticas a partir de EEG de superficie},
  school = {Universidad Internacional de La Rioja (UNIR)},
  year   = {2026},
  type   = {Trabajo Fin de Máster},
}
```

---

## Licencia

Este código se distribuye con fines académicos como anexo del TFM.

