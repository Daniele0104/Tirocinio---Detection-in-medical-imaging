import torch
import numpy as np
import os

path = "folds_k3/fold_2/test_results/eval.pth"
data = torch.load(path, map_location="cpu")

precision = data["precision"]  # [TxRxKxAxM]
recall = data["recall"]        # [TxKxAxM]
counts = data.get("counts", None)

T, R, K, A, M = precision.shape
print(f"IoU thresholds: {T}, recall steps: {R}, classes: {K}")

# Media su aree e maxDets → [TxRxK], [TxK]
precision = precision.mean(axis=(3, 4))
recall = recall.mean(axis=(2, 3))

# Pulizia: sostituisci valori -1 con NaN per evitarli nelle medie
precision = np.where(precision < 0, np.nan, precision)
recall = np.where(recall < 0, np.nan, recall)

# Conta ground truth per classe
if isinstance(counts, (list, tuple, np.ndarray)):
    n_gt = np.array(counts)
    if n_gt.size < K:
        n_gt = np.pad(n_gt, (0, K - n_gt.size), constant_values=1)
else:
    n_gt = np.ones(K)

tp, fp, fn = [], [], []

for k in range(K):
    rec_vals = recall[:, k]
    rec_vals = rec_vals[~np.isnan(rec_vals)]
    rec = np.nanmean(rec_vals) if rec_vals.size > 0 else 0.0

    prec_vals = precision[:, :, k]
    prec_vals = prec_vals[~np.isnan(prec_vals)]
    prec = np.nanmean(prec_vals) if prec_vals.size > 0 else 0.0

    n = max(n_gt[k], 1)
    TP = max(rec * n, 0)
    FP = TP * (1 / prec - 1) if prec > 0 else 0
    FN = max(n - TP, 0)

    tp.append(TP)
    fp.append(FP)
    fn.append(FN)

# --- stampa a schermo ---
for k, (T, Fp, Fn) in enumerate(zip(tp, fp, fn), start=1):
    print(f"Class {k:02d}: TP={T:.2f}, FP={Fp:.2f}, FN={Fn:.2f}")

# --- salvataggio su file ---
output_dir = os.path.dirname(path)
output_file = os.path.join(output_dir, "metrics_summary.txt")

with open(output_file, "w") as f:
    f.write(f"IoU thresholds: {T}, recall steps: {R}, classes: {K}\n\n")
    for k, (T, Fp, Fn) in enumerate(zip(tp, fp, fn), start=1):
        prec = T / (T + Fp) if (T + Fp) > 0 else 0.0
        rec = T / (T + Fn) if (T + Fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f.write(
            f"Class {k:02d}: TP={T:.2f}, FP={Fp:.2f}, FN={Fn:.2f}, "
            f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}\n"
        )

print(f"\n✅ Risultati salvati in: {output_file}")
