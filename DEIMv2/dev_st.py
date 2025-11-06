"""import json
import numpy as np

# --------------------------------------------------
# 1️⃣ File JSON dei 3 fold
# --------------------------------------------------
fold_files = ["DEIMv2\\folds_k3\\fold_0\\metrics_fold_0.json", "DEIMv2\\folds_k3\\fold_1\\metrics_fold_1.json", "DEIMv2\\folds_k3\\fold_2\\metrics_fold_2.json"]

# Nomi metriche COCO
metric_names = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"
]

# --------------------------------------------------
# 2️⃣ Caricamento dati
# --------------------------------------------------
metrics_per_fold = []

for f in fold_files:
    with open(f, "r") as file:
        data = json.load(file)
    # Prendiamo i risultati dell'ultima epoch (puoi cambiarlo se vuoi la media delle epoch)
    last_epoch = data[-1]
    metrics_per_fold.append(last_epoch["test_coco_eval_bbox"])

# Convertiamo in array numpy: (num_fold, num_metriche)
metrics_array = np.array(metrics_per_fold)

# --------------------------------------------------
# 3️⃣ Calcolo media e deviazione standard
# --------------------------------------------------
metrics_mean = np.mean(metrics_array, axis=0)
metrics_std  = np.std(metrics_array, axis=0)

# --------------------------------------------------
# 4️⃣ Stampa risultati in formato tabellare
# --------------------------------------------------
print("\n📊 Risultati COCO (media ± std su 3 fold):\n")
print(f"{'Metrica':<15} {'Media':>10} {'Dev.Std':>10}")
print("-" * 40)

for name, mean, std in zip(metric_names, metrics_mean, metrics_std):
    print(f"{name:<15} {mean:>10.6f} {std:>10.6f}")

# --------------------------------------------------
# 5️⃣ (Opzionale) Salvataggio su file di testo
# --------------------------------------------------
with open("metrics_summary.txt", "w") as out:
    out.write("Risultati COCO (media ± std su 3 fold)\n\n")
    out.write(f"{'Metrica':<15} {'Media':>10} {'Dev.Std':>10}\n")
    out.write("-" * 40 + "\n")
    for name, mean, std in zip(metric_names, metrics_mean, metrics_std):
        out.write(f"{name:<15} {mean:>10.6f} {std:>10.6f}\n")

print("\n✅ File 'metrics_summary.txt' salvato.")
"""