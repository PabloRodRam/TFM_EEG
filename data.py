"""
Carga y parsing de datos EEG desde la base de datos CHB-MIT.

Funciones para:
  - Parsear los ficheros RECORDS-WITH-SEIZURES de CHB-MIT.
  - Calcular tiempos absolutos de crisis.
  - Seleccionar *leading seizures* (crisis separadas >= 60 min).
  - Cargar señal continua EDF seleccionando 23 canales bipolares estándar.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np

from .config import (
    LEADING_GAP_MIN,
    MIN_LEADING,
    RAW_DIR,
    STANDARD_CHANNELS,
)


# ================================================================
# Parsing del fichero summary
# ================================================================

def parse_summary(summary_path: Path) -> list[dict]:
    """Parsea el fichero ``*-summary.txt`` de CHB-MIT para un sujeto.

    Devuelve una lista de registros con campos:
        - subject (str)
        - file (str)
        - seizures: list[(start_sec, end_sec)]
    """
    records: list[dict] = []
    current_file = None
    n_seizures = 0
    seizures: list[tuple[float, float]] = []
    subject = summary_path.parent.name

    with open(summary_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                if current_file is not None:
                    records.append({
                        "subject": subject,
                        "file": current_file,
                        "seizures": seizures,
                    })
                current_file = line.split(":")[-1].strip()
                n_seizures = 0
                seizures = []
            elif "Number of Seizures in File:" in line:
                parts = line.split(":")
                n_seizures = int(parts[-1].strip())
            elif "Seizure" in line and "Start" in line:
                val = re.findall(r"\d+", line)
                if val:
                    seizures.append((float(val[0]), 0.0))
            elif "Seizure" in line and "End" in line:
                val = re.findall(r"\d+", line)
                if val and seizures:
                    start = seizures[-1][0]
                    seizures[-1] = (start, float(val[0]))

    if current_file is not None:
        records.append({
            "subject": subject,
            "file": current_file,
            "seizures": seizures,
        })

    return records


# ================================================================
# Tiempos absolutos de crisis
# ================================================================

def compute_absolute_seizure_times(records: list[dict]) -> list[dict]:
    """Calcula tiempo absoluto (s desde inicio de grabación) de cada crisis."""
    from datetime import datetime, timedelta

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_subject[rec["subject"]].append(rec)

    seizure_abs: list[dict] = []

    for subj, recs in sorted(by_subject.items()):
        file_info = []
        for rec in recs:
            edf_path = RAW_DIR / subj / rec["file"]
            if not edf_path.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=False,
                                          verbose=False)
                start_dt = raw.info["meas_date"]
                duration = raw.n_times / raw.info["sfreq"]
                file_info.append({
                    "rec": rec,
                    "start_dt": start_dt,
                    "duration": duration,
                    "path": edf_path,
                })
            except Exception:
                continue

        if not file_info:
            continue

        file_info.sort(key=lambda x: x["start_dt"])
        ref_dt = file_info[0]["start_dt"]

        for fi in file_info:
            offset_sec = (fi["start_dt"] - ref_dt).total_seconds()
            for sz_start, sz_end in fi["rec"]["seizures"]:
                seizure_abs.append({
                    "subject": subj,
                    "file": fi["rec"]["file"],
                    "abs_start": offset_sec + sz_start,
                    "abs_end": offset_sec + sz_end,
                    "file_offset": offset_sec,
                    "file_duration": fi["duration"],
                })

    return seizure_abs


# ================================================================
# Selección de leading seizures
# ================================================================

def select_leading_seizures(
    seizure_abs: list[dict],
    gap_min: int = LEADING_GAP_MIN,
) -> dict[str, list[dict]]:
    """Selecciona *leading seizures* (separadas >= *gap_min* minutos)."""
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for s in seizure_abs:
        by_subj[s["subject"]].append(s)

    leading: dict[str, list[dict]] = {}
    for subj, szs in sorted(by_subj.items()):
        szs_sorted = sorted(szs, key=lambda x: x["abs_start"])
        selected = [szs_sorted[0]]
        for s in szs_sorted[1:]:
            if (s["abs_start"] - selected[-1]["abs_end"]) >= gap_min * 60:
                selected.append(s)
        leading[subj] = selected

    return leading


def get_valid_subjects(
    leading: dict[str, list[dict]],
    min_leading: int = MIN_LEADING,
) -> dict[str, list[dict]]:
    """Filtra sujetos con al menos *min_leading* leading seizures."""
    return {s: szs for s, szs in leading.items() if len(szs) >= min_leading}


# ================================================================
# Carga de señal continua
# ================================================================

def _build_channel_picks(raw_ch_names: list[str]) -> list[str] | None:
    """Construye lista de 23 canales del EDF (gestiona duplicados MNE)."""
    base_to_real: dict[str, list[str]] = defaultdict(list)
    for ch in raw_ch_names:
        base = re.sub(r"-(\d+)$", "", ch)
        base_to_real[base].append(ch)

    picks: list[str] = []
    base_usage: dict[str, int] = defaultdict(int)

    for std_ch in STANDARD_CHANNELS:
        std_upper = std_ch.upper()
        usage_idx = base_usage[std_upper]

        if std_upper in base_to_real:
            reals = base_to_real[std_upper]
            if usage_idx < len(reals):
                picks.append(reals[usage_idx])
                base_usage[std_upper] += 1
            else:
                picks.append(reals[-1])
        else:
            return None

    return picks


def load_subject_continuous(
    subject: str,
    fs: int = 256,
) -> tuple[np.ndarray, float]:
    """Carga todos los EDF de un sujeto y concatena en orden cronológico.

    Returns
    -------
    data : ndarray (n_samples, 23)
        Señal EEG continua.
    ref_ts : float
        Timestamp de referencia del primer fichero.
    """
    subj_dir = RAW_DIR / subject
    edfs = sorted(subj_dir.glob("*.edf"))

    file_data: list[tuple] = []
    skipped = 0

    for edf_path in edfs:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True,
                                      verbose=False)
        except Exception:
            skipped += 1
            continue

        # Renombrar canales a mayúsculas
        rename_map = {}
        for ch in raw.ch_names:
            new_name = ch.upper().replace(" ", "")
            if new_name != ch:
                rename_map[ch] = new_name
        if rename_map:
            raw.rename_channels(rename_map)

        picks = _build_channel_picks(raw.ch_names)
        if picks is None:
            skipped += 1
            continue

        try:
            raw.pick_channels(picks, ordered=True)
        except Exception:
            skipped += 1
            continue

        start_dt = raw.info["meas_date"]
        arr = raw.get_data().T
        file_data.append((start_dt, arr))

    if not file_data:
        return np.array([]), 0.0

    if skipped > 0:
        print(f"  Info: {skipped} ficheros EDF omitidos")

    file_data.sort(key=lambda x: x[0])
    ref_ts = file_data[0][0].timestamp()
    data = np.concatenate([fd[1] for fd in file_data], axis=0)
    return data.astype(np.float32), ref_ts
