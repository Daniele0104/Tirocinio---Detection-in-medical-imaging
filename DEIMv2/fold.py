import os
import json
import subprocess
import platform
import sklearn
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_annotations(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    k = 3
    ann_path = "C:/Tirocinio/DEIMv2/dataset/annotations/trainval_no_rbc.json"
    ann = load_coco_annotations(ann_path)
    images = ann["images"]
    anns = ann["annotations"]

    # etichette per stratificazione
    image_to_labels = {}
    for a in anns:
        image_to_labels.setdefault(a["image_id"], []).append(a["category_id"])
    
    # Assicurati che tutte le immagini abbiano almeno un'etichetta per la stratificazione
    labels = []
    valid_images = []
    for i in images:
        if i["id"] in image_to_labels:
            labels.append(image_to_labels[i["id"]][0])
            valid_images.append(i)
        # else:
            # print(f"Warning: Image {i['id']} has no annotations, skipping.")
    
    images = valid_images # Usa solo immagini con etichette

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n=== FOLD {fold+1}/{k} ===")

        train_images = [images[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]

        train_ids = {img["id"] for img in train_images}
        val_ids = {img["id"] for img in val_images}

        train_anns = [a for a in anns if a["image_id"] in train_ids]
        val_anns = [a for a in anns if a["image_id"] in val_ids]

        train_data = {
            "images": train_images,
            "annotations": train_anns,
            "categories": ann["categories"]
        }
        val_data = {
            "images": val_images,
            "annotations": val_anns,
            "categories": ann["categories"]
        }

        fold_dir = os.path.join(f"C:/Tirocinio/DEIMv2/folds_k{k}", f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        train_json = os.path.join(fold_dir, "train.json")
        val_json = os.path.join(fold_dir, "val.json")

        save_coco_annotations(train_data, train_json)
        save_coco_annotations(val_data, val_json)

        # TRAINING 
        print(f"\nTraining fold {fold}...\n")
        train_cmd = [
            "python", "train.py",
            "-c", "C:/Tirocinio/DEIMv2/configs/deimv2/deimv2_hgnetv2_atto_coco.yml",
            "--use-amp",
            "--seed", "0",
            "--device", "cuda:0",
            "--output-dir", fold_dir,
            # --- MODIFICA CORRETTA ---
            "-u", f"train_dataloader.dataset.ann_file={train_json}", f"val_dataloader.dataset.ann_file={val_json}"
        ]
        subprocess.run(train_cmd, check=True)

        # TEST
        print(f"\nTest fold {fold}\n")
        best_model = os.path.join(fold_dir, "best_stg2.pth")
        test_results_dir = os.path.join(fold_dir, "test_results")
        os.makedirs(test_results_dir, exist_ok=True)

        test_cmd = [
            "python", "train.py",
            "-c", "C:/Tirocinio/DEIMv2/configs/deimv2/deimv2_hgnetv2_atto_coco.yml",
            "--test-only",
            "--device", "cuda:0",
            # --- MODIFICA CORRETTA ---
            "-u", f"val_dataloader.dataset.ann_file={val_json}",
            "--resume", best_model,
            "--output-dir", test_results_dir
        ]
        subprocess.run(test_cmd, check=True)

    # METADATA
    metadata = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "random_state": 42,
        "k_folds": k,
        "config_file": "C:/Tirocinio/DEIMv2/configs/deimv2/deimv2_hgnetv2_atto_coco.yml"
    }

    save_coco_annotations(metadata, f"C:/Tirocinio/DEIMv2/folds_k{k}/metadata.json")
    print("\nTutti i fold completati correttamente!")

if __name__ == "__main__":
    main()