import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.distributed as dist
from ..precision import get_autocast, get_input_dtype
from .metrics import auroc, accuracy_and_f1
import logging


def unwrap_model(model):
    """Unwrap DistributedDataParallel if needed."""
    return model.module if hasattr(model, "module") else model


def run(X, y, X_test, y_test):
    _, num_classes = y.shape

    preds = np.zeros_like(y_test)
    for i in range(num_classes):
        lr_model = LogisticRegression()
        lr_model.fit(X, y[:, i])
        preds[:, i] = lr_model.predict_proba(X_test)[:, 1]

    acc, f1, _, _ = accuracy_and_f1(y_test, preds)
    auc, _ = auroc(y_test, preds)
    return acc, f1, auc


def linear_probe_eval(model, train_data, test_data, args, dataset=""):
    # ============================================================
    # 只在 rank 0 上执行（保持你原来的逻辑）
    # ============================================================
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return {}

    logging.info(f'Starting linear-probe {dataset}.')
    metrics = {}

    all_train_ecg_features = []
    all_train_labels = []
    all_test_ecg_features = []
    all_test_labels = []

    device = args.device
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # ✅ unwrap，一次到位，彻底绕开 DDP forward
    model_ = unwrap_model(model)
    model_.eval()

    with torch.no_grad():
        # -------- Train features --------
        for i, batch in enumerate(train_data):
            ecgs, targets = batch
            all_train_labels.append(targets)

            ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
            with autocast():
                # ✅ 核心修改：只走 encoder
                ecg_features = model_.encode_ecg(ecgs)
                all_train_ecg_features.append(ecg_features.cpu())

        all_train_ecg_features = torch.cat(all_train_ecg_features)
        all_train_labels = torch.cat(all_train_labels)

        # -------- Test features --------
        for i, batch in enumerate(test_data):
            ecgs, targets = batch
            all_test_labels.append(targets)

            ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
            with autocast():
                # ✅ 同样只走 encoder
                ecg_features = model_.encode_ecg(ecgs)
                all_test_ecg_features.append(ecg_features.cpu())

        all_test_ecg_features = torch.cat(all_test_ecg_features)
        all_test_labels = torch.cat(all_test_labels)

        # -------- Linear probe (sklearn) --------
        acc, f1, roc_auc = run(
            all_train_ecg_features,
            all_train_labels,
            all_test_ecg_features,
            all_test_labels
        )

        metrics[f"{dataset}-linear-probe-val-acc"] = acc
        metrics[f"{dataset}-linear-probe-val-f1-score"] = f1
        metrics[f"{dataset}-linear-probe-val-auc"] = roc_auc

    return metrics
