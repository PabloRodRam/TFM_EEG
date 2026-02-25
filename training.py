"""
Funciones de entrenamiento, split de datos y calibración.

Incluye:
  - Split cronológico por crisis y split proporcional.
  - Entrenamiento con 10-fold CV (LSTM, Transformer, CNN-LSTM).
  - Entrenamiento SVM (v1 gamma y v4 broadband).
  - Calibración de firing_k y threshold.
  - LOSO cross-subject data loader (Flujo).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from .config import DEVICE, FS, N_CHANNELS, WIN_SEC
from .models import SeizureCNNLSTM, SeizureLSTM, SeizureTransformer


# ================================================================
# Splits de datos
# ================================================================

def split_train_test_by_seizure(
    subject: str,
    seizures_abs: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    n_train_seizures: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split cronológico por crisis (Wu et al. 2023).

    Las primeras *n_train_seizures* leading seizures van a train,
    el resto a test.
    """
    n_sz = len(seizures_abs)
    n_train_sz = min(n_train_seizures, n_sz - 1)

    pre_idx = np.where(y == 1)[0]
    inter_idx = np.where(y == 0)[0]

    if len(pre_idx) == 0:
        return X, y, X[:0], y[:0]

    n_pre_train = int(len(pre_idx) * n_train_sz / n_sz)
    n_pre_train = max(1, min(n_pre_train, len(pre_idx) - 1))

    train_pre = pre_idx[:n_pre_train]
    test_pre = pre_idx[n_pre_train:]

    if len(train_pre) > 0 and len(test_pre) > 0:
        boundary = (train_pre[-1] + test_pre[0]) // 2
        train_inter = inter_idx[inter_idx <= boundary]
        test_inter = inter_idx[inter_idx > boundary]
    else:
        train_inter = inter_idx
        test_inter = np.array([], dtype=int)

    train_idx = np.sort(np.concatenate([train_pre, train_inter]))
    test_idx = np.sort(np.concatenate([test_pre, test_inter]))

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ================================================================
# Utilidades de entrenamiento
# ================================================================

def _compute_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Calcula pesos de clase balanceados para CrossEntropyLoss."""
    classes = np.unique(y)
    if len(classes) < 2:
        return torch.ones(2, dtype=torch.float32).to(device)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(cw, dtype=torch.float32).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Entrena un epoch completo.  Devuelve (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evalúa el modelo.  Devuelve (loss, acc, preds, labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


# ================================================================
# Entrenamiento LSTM v1 (10-fold CV)
# ================================================================

def train_subject_model(
    subject: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = 2,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    n_folds: int = 10,
    device: torch.device = DEVICE,
) -> dict:
    """Entrena modelo LSTM v1 con 10-fold CV interno para un sujeto."""
    print(f"\n  Entrenando LSTM para {subject}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    class_weights = _compute_class_weights(y_train, device)

    # 10-fold CV para determinar mejor nº de epochs
    n_splits = min(n_folds, max(2, len(X_train) // 10))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_epoch_per_fold: list[int] = []

    for _fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_train[tr_idx], dtype=torch.long)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train[val_idx], dtype=torch.long)

        tr_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size,
        )

        model = SeizureLSTM(
            input_size=N_CHANNELS, hidden_size=128, n_classes=n_classes,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        fold_cw = _compute_class_weights(y_train[tr_idx], device)
        criterion = nn.CrossEntropyLoss(weight=fold_cw)

        best_val_loss = float("inf")
        best_ep = 0
        wait = 0

        for ep in range(n_epochs):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            val_loss, *_ = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ep = ep + 1
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        best_epoch_per_fold.append(best_ep)
        del model, optimizer

    optimal_epochs = max(int(np.median(best_epoch_per_fold)), 5)
    print(f"  CV epochs: {best_epoch_per_fold} -> mediana={optimal_epochs}")

    # Entrenamiento final
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_te_t, y_te_t), batch_size=batch_size,
    )

    model = SeizureLSTM(
        input_size=N_CHANNELS, hidden_size=128, n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for ep in range(optimal_epochs):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        if (ep + 1) % 10 == 0 or ep == optimal_epochs - 1:
            print(f"    Epoch {ep+1}/{optimal_epochs}: loss={t_loss:.4f}, acc={t_acc:.4f}")

    # Evaluación test
    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device,
    )

    sen = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(test_labels, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)

    print(f"  Resultados: Acc={test_acc:.4f} Sen={sen:.4f} Spec={spec:.4f} F1={f1:.4f}")

    return {
        "model": model,
        "subject": subject,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "optimal_epochs": optimal_epochs,
        "test_acc": test_acc,
        "sensitivity": sen,
        "specificity": spec,
        "f1": f1,
        "test_preds": test_preds,
        "test_labels": test_labels,
    }


# ================================================================
# Entrenamiento CNN-Transformer v1 (10-fold CV)
# ================================================================

def train_subject_transformer(
    subject: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_freq: int = 129,
    n_classes: int = 2,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    n_folds: int = 10,
    device: torch.device = DEVICE,
) -> dict:
    """Entrena SeizureTransformer (v1) con 10-fold CV."""
    print(f"\n  Entrenando Transformer para {subject}")

    class_weights = _compute_class_weights(y_train, device)
    n_splits = min(n_folds, max(2, len(X_train) // 10))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_epoch_per_fold: list[int] = []

    for _fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_train[tr_idx], dtype=torch.long)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train[val_idx], dtype=torch.long)

        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        model = SeizureTransformer(
            n_channels=N_CHANNELS, n_freq=n_freq,
            d_model=128, nhead=8, num_layers=2, n_classes=n_classes, dropout=0.3,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        fold_cw = _compute_class_weights(y_train[tr_idx], device)
        criterion = nn.CrossEntropyLoss(weight=fold_cw)

        best_val_loss = float("inf")
        best_ep = 0
        wait = 0

        for ep in range(n_epochs):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            val_loss, *_ = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ep = ep + 1
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        best_epoch_per_fold.append(best_ep)
        del model, optimizer
        torch.cuda.empty_cache()

    optimal_epochs = max(int(np.median(best_epoch_per_fold)), 5)
    print(f"  CV epochs: {best_epoch_per_fold} -> mediana={optimal_epochs}")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    model = SeizureTransformer(
        n_channels=N_CHANNELS, n_freq=n_freq,
        d_model=128, nhead=8, num_layers=2, n_classes=n_classes, dropout=0.3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for ep in range(optimal_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if (ep + 1) % 10 == 0 or ep == optimal_epochs - 1:
            print(f"    Epoch {ep+1}/{optimal_epochs}: loss={t_loss:.4f}, acc={t_acc:.4f}")

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device,
    )

    sen = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(test_labels, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)

    print(f"  Resultados: Acc={test_acc:.4f} Sen={sen:.4f} Spec={spec:.4f} F1={f1:.4f}")

    return {
        "model": model,
        "subject": subject,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "optimal_epochs": optimal_epochs,
        "test_acc": test_acc,
        "sensitivity": sen,
        "specificity": spec,
        "f1": f1,
        "test_preds": test_preds,
        "test_labels": test_labels,
    }


# ================================================================
# Entrenamiento SVM v1 (gamma)
# ================================================================

def train_subject_svm(
    subject: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    C: float = 1.0,
    gamma_param: str = "scale",
    kernel: str = "rbf",
) -> dict:
    """Entrena SVM RBF v1 con z-score + PCA opcional."""
    print(f"\n  Entrenando SVM para {subject}")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    pca = None
    if use_pca and X_train_s.shape[1] > 50:
        pca = PCA(n_components=pca_variance, random_state=42)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)
        print(f"  PCA: {X_train.shape[1]} -> {X_train_s.shape[1]} componentes")

    svm = SVC(
        kernel=kernel, C=C, gamma=gamma_param,
        class_weight="balanced", random_state=42,
    )
    svm.fit(X_train_s, y_train)

    test_preds = svm.predict(X_test_s)
    test_acc = accuracy_score(y_test, test_preds)
    sen = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(y_test, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)

    print(f"  Resultados: Acc={test_acc:.4f} Sen={sen:.4f} Spec={spec:.4f} F1={f1:.4f}")

    return {
        "subject": subject,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_acc": test_acc,
        "sensitivity": sen,
        "specificity": spec,
        "f1": f1,
        "test_preds": test_preds,
        "test_labels": y_test,
        "svm_model": svm,
        "scaler": scaler,
        "pca": pca,
    }


# ================================================================
# Entrenamiento CNN-LSTM v4 (10-fold CV)
# ================================================================

def train_cnnlstm_v4(
    subject: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_freq: int = 129,
    n_classes: int = 2,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    n_folds: int = 10,
    device: torch.device = DEVICE,
) -> dict:
    """Entrena CNN-LSTM v4 con 10-fold CV sobre espectrogramas broadband."""
    print(f"\n  [CNN-LSTM] Entrenando para {subject}")

    class_weights = _compute_class_weights(y_train, device)
    n_splits = min(n_folds, max(2, len(X_train) // 10))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_epoch_per_fold: list[int] = []

    for _fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_train[tr_idx], dtype=torch.long)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train[val_idx], dtype=torch.long)

        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        model = SeizureCNNLSTM(
            n_channels=N_CHANNELS, n_freq=n_freq, d_cnn=64, lstm_hidden=128,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        fold_cw = _compute_class_weights(y_train[tr_idx], device)
        criterion = nn.CrossEntropyLoss(weight=fold_cw)

        best_val_loss = float("inf")
        best_ep = 0
        wait = 0

        for ep in range(n_epochs):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            val_loss, *_ = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ep = ep + 1
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        best_epoch_per_fold.append(best_ep)
        del model, optimizer
        torch.cuda.empty_cache()

    optimal_epochs = max(int(np.median(best_epoch_per_fold)), 5)
    print(f"  CV epochs: {best_epoch_per_fold} -> mediana={optimal_epochs}")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    model = SeizureCNNLSTM(
        n_channels=N_CHANNELS, n_freq=n_freq, d_cnn=64, lstm_hidden=128,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for ep in range(optimal_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if (ep + 1) % 10 == 0 or ep == optimal_epochs - 1:
            print(f"    Epoch {ep+1}/{optimal_epochs}: loss={t_loss:.4f}, acc={t_acc:.4f}")

    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device,
    )

    sen = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(test_labels, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)

    print(f"  Resultados: Acc={test_acc:.4f} Sen={sen:.4f} Spec={spec:.4f} F1={f1:.4f}")

    return {
        "model": model,
        "model_type": "CNN-LSTM",
        "subject": subject,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "optimal_epochs": optimal_epochs,
        "test_acc": test_acc,
        "sensitivity": sen,
        "specificity": spec,
        "f1": f1,
        "test_preds": test_preds,
        "test_labels": test_labels,
    }


# ================================================================
# Entrenamiento SVM v4 (broadband)
# ================================================================

def train_svm_v4(
    subject: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Entrena SVM RBF v4 sobre features 345-dim de espectrogramas broadband."""
    print(f"\n  [SVM] Entrenando para {subject}")

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    n_components = min(50, X_tr_scaled.shape[1], X_tr_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_te_pca = pca.transform(X_te_scaled)
    print(
        f"  PCA: {X_tr_scaled.shape[1]} -> {n_components} componentes "
        f"(var={pca.explained_variance_ratio_.sum():.2%})"
    )

    svm_model = SVC(
        kernel="rbf", C=10.0, gamma="scale",
        class_weight="balanced", random_state=42,
    )
    svm_model.fit(X_tr_pca, y_train)

    test_preds = svm_model.predict(X_te_pca)
    test_acc = accuracy_score(y_test, test_preds)
    sen = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(y_test, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)

    print(f"  Resultados: Acc={test_acc:.4f} Sen={sen:.4f} Spec={spec:.4f} F1={f1:.4f}")

    return {
        "model": svm_model,
        "scaler": scaler,
        "pca": pca,
        "model_type": "SVM",
        "subject": subject,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_acc": test_acc,
        "sensitivity": sen,
        "specificity": spec,
        "f1": f1,
        "test_preds": test_preds,
        "test_labels": y_test,
    }


# ================================================================
# Entrenamiento CNN-Transformer v4 (broadband, reutiliza SeizureTransformer)
# ================================================================

def train_transformer_v4(
    subject: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_freq: int = 129,
    n_classes: int = 2,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    n_folds: int = 10,
    device: torch.device = DEVICE,
) -> dict:
    """Entrena CNN-Transformer v4 (SeizureTransformer) sobre espectrogramas broadband."""
    print(f"\n  [Transformer] Entrenando para {subject}")

    class_weights = _compute_class_weights(y_train, device)
    n_splits = min(n_folds, max(2, len(X_train) // 10))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_epoch_per_fold: list[int] = []

    for _fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_train[tr_idx], dtype=torch.long)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train[val_idx], dtype=torch.long)

        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        model = SeizureTransformer(
            n_channels=N_CHANNELS, n_freq=n_freq,
            d_model=128, nhead=8, num_layers=2, dim_feedforward=256,
            n_classes=n_classes, dropout=0.3,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        fold_cw = _compute_class_weights(y_train[tr_idx], device)
        criterion = nn.CrossEntropyLoss(weight=fold_cw)

        best_val_loss = float("inf")
        best_ep = 0
        wait = 0

        for ep in range(n_epochs):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            val_loss, *_ = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ep = ep + 1
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        best_epoch_per_fold.append(best_ep)
        del model, optimizer
        torch.cuda.empty_cache()

    optimal_epochs = max(int(np.median(best_epoch_per_fold)), 5)
    print(f"  CV epochs: {best_epoch_per_fold} -> mediana={optimal_epochs}")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    model = SeizureTransformer(
        n_channels=N_CHANNELS, n_freq=n_freq,
        d_model=128, nhead=8, num_layers=2, dim_feedforward=256,
        n_classes=n_classes, dropout=0.3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for ep in range(optimal_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if (ep + 1) % 10 == 0 or ep == optimal_epochs - 1:
            print(f"    Epoch {ep+1}/{optimal_epochs}: loss={t_loss:.4f}, acc={t_acc:.4f}")

    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device,
    )

    sen = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(test_labels, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)

    print(f"  Resultados: Acc={test_acc:.4f} Sen={sen:.4f} Spec={spec:.4f} F1={f1:.4f}")

    return {
        "model": model,
        "model_type": "Transformer",
        "subject": subject,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "optimal_epochs": optimal_epochs,
        "test_acc": test_acc,
        "sensitivity": sen,
        "specificity": spec,
        "f1": f1,
        "test_preds": test_preds,
        "test_labels": test_labels,
    }


# ================================================================
# Calibración de firing_k (v4)
# ================================================================

def calibrate_firing_k(
    preds_5s: np.ndarray,
    labels_5s: np.ndarray,
    firing_n: int = 20,
    win_sec: int = WIN_SEC,
    far_penalty: float = 0.5,
) -> int:
    """Calibra *k* óptimo del firing rule en datos de validación.

    Maximiza ``composite = Event_Sensitivity − penalty × FAR/h``.
    """
    from .evaluation import apply_firing_rule

    best_score = -np.inf
    best_k = firing_n // 2

    for k in range(max(1, firing_n // 4), firing_n + 1):
        fired = apply_firing_rule(preds_5s, k=k, n=firing_n)

        # Sensitivity proxy
        pre_mask = labels_5s == 1
        sen_proxy = fired[pre_mask].mean() if pre_mask.sum() > 0 else 0.0

        # FAR proxy
        inter_mask = labels_5s == 0
        inter_fired = fired[inter_mask]
        inter_hours = inter_mask.sum() * win_sec / 3600
        if len(inter_fired) > 1 and inter_hours > 0.01:
            transitions = np.diff(inter_fired)
            n_fa = int((transitions == 1).sum())
            if inter_fired[0] == 1:
                n_fa += 1
            far_est = n_fa / inter_hours
        else:
            far_est = 0.0

        composite = sen_proxy - far_penalty * far_est
        if composite > best_score:
            best_score = composite
            best_k = k

    return best_k


# ================================================================
# LOSO cross-subject data loader (Flujo)
# ================================================================

def get_loso_data(
    patient_ids: list[str],
    test_pid: str,
    preprocessed_dir,
    train_ratio_ns: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga datos LOSO: todos los pacientes excepto *test_pid* → train.

    Parameters
    ----------
    patient_ids : list[str]
        Lista de IDs de pacientes.
    test_pid : str
        ID del paciente de test.
    preprocessed_dir : Path
        Directorio con ficheros ``{pid}_preprocessed.npz``.
    train_ratio_ns : float
        Ratio de undersampling interictal en train.
    seed : int
        Semilla para undersampling.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    X_train_list, y_train_list = [], []
    X_test, y_test = None, None

    for pid in patient_ids:
        npz_path = preprocessed_dir / f"{pid}_preprocessed.npz"
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))
        X_p = data["X"]
        y_p = data["y"]

        if pid == test_pid:
            X_test = X_p
            y_test = y_p
        else:
            X_train_list.append(X_p)
            y_train_list.append(y_p)

    if not X_train_list or X_test is None:
        raise ValueError(f"Datos insuficientes para LOSO con test={test_pid}")

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Undersampling interictal en train
    n_pos = int((y_train == 1).sum())
    n_ns_target = int(n_pos * train_ratio_ns)
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]

    if len(idx_neg) > n_ns_target:
        idx_neg = rng.choice(idx_neg, size=n_ns_target, replace=False)

    idx_all = np.sort(np.concatenate([idx_pos, idx_neg]))
    X_train = X_train[idx_all]
    y_train = y_train[idx_all]

    return X_train, y_train, X_test, y_test
