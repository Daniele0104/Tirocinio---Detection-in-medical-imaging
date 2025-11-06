import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

# === CONFIG ===
BASE_DIR = "C:/Tirocinio/DEIMv2/folds_k3"
OUTPUT_DIR = os.path.join(BASE_DIR, "risultati")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_metrics_summary(path):
    """Legge TP, FP, FN per classe dal file metrics_summary.txt"""
    TP, FP, FN = [], [], []
    with open(path, "r") as f:
        for line in f:
            if "Class" in line and "TP=" in line:
                parts = line.strip().split(",")
                tp = float(parts[0].split("TP=")[1])
                fp = float(parts[1].split("FP=")[1])
                fn = float(parts[2].split("FN=")[1])
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
    return np.array(TP), np.array(FP), np.array(FN)

def compute_metrics(TP, FP, FN):
    """Calcola metriche per ogni classe"""
    eps = 1e-10
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    specificity = TP / (TP + FP + FN + eps)  # proxy, in OD non c'è TN diretto
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = TP / (TP + FP + FN + eps)
    balanced_acc = (recall + specificity) / 2

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc
    }

def aggregate_global_metrics(per_class_metrics):
    """Aggrega globalmente in modo macro, micro e weighted"""
    metrics = {}
    weights = per_class_metrics["TP"] + per_class_metrics["FN"]
    weights = weights / np.sum(weights)  # frequenze relative

    for key, values in per_class_metrics.items():
        if key in ["TP", "FP", "FN"]:
            continue
        vals = np.array(values)
        metrics[f"{key}_macro"] = np.nanmean(vals)
        metrics[f"{key}_micro"] = np.nansum(vals * weights)
        metrics[f"{key}_weighted"] = np.nansum(vals * weights)

    return metrics

# === MAIN ===
all_fold_metrics = []

for fold in range(3):
    path = os.path.join(BASE_DIR, f"fold_{fold}", "test_results", "metrics_summary.txt")
    if not os.path.exists(path):
        print(f"❌ Manca {path}")
        continue

    TP, FP, FN = parse_metrics_summary(path)
    m = compute_metrics(TP, FP, FN)
    m["TP"], m["FP"], m["FN"] = TP, FP, FN
    agg = aggregate_global_metrics(m)
    all_fold_metrics.append(agg)

# === MEDIA + DEVIAZIONE STANDARD ===
keys = all_fold_metrics[0].keys()
mean_metrics = {k: np.mean([f[k] for f in all_fold_metrics]) for k in keys}
std_metrics = {k: np.std([f[k] for f in all_fold_metrics]) for k in keys}

# === SALVA RISULTATI ===
output_file = os.path.join(OUTPUT_DIR, "global_metrics.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== METRICHE GLOBALI SU K-FOLD ===\n\n")
    for k in keys:
        f.write(f"{k:25s}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}\n")

print(f"\n✅ Metriche globali salvate in: {output_file}")
