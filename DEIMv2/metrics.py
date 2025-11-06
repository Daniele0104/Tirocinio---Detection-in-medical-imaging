"""import json
import os

base_dir = r"C:/Tirocinio/DEIMv2/folds_k3"
num_folds = 3 

for fold_idx in range(num_folds):
    fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
    log_path = os.path.join(fold_dir, "log.txt")
    out_path = os.path.join(fold_dir, f"metrics_fold_{fold_idx}.json")

    if not os.path.exists(log_path):
        print(f"Nessun log trovato per fold {fold_idx}")
        continue

    metrics = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                metrics.append({
                    "epoch": data.get("epoch"),
                    "train_loss": data.get("train_loss"),
                    "test_coco_eval_bbox": data.get("test_coco_eval_bbox")
                })
            except json.JSONDecodeError:
                print(f"⚠️ Errore nel parsing della riga nel fold {fold_idx}:")
                print(line[:200])
    
    if metrics:
        with open(out_path, "w") as out:
            json.dump(metrics, out, indent=2)
        print(f"Salvato: {out_path} ({len(metrics)} epoche trovate)")
    else:
        print(f"Nessuna epoca valida trovata per fold {fold_idx}")"""
