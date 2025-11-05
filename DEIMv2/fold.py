import os
import json
from sklearn.model_selection import StratifiedKFold
import subprocess
import platform, sklearn, torch, numpy

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_annotations(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def main():
    k=3
    ann = "C:/Tirocinio/DEIMv2/dataset/annotations/trainval.json"
    ann = load_coco_annotations(ann)
    images = ann["images"]
    anns = ann["annotations"]

    image_to_labels = {}
    for a in anns:
        image_to_labels.setdefault(a["image_id"], []).append(a["category_id"])
    labels = [image_to_labels[i["id"]][0] for i in images]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n=== Fold {fold+1}/{k} ===")

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

        cmd = [
            "python", "train.py",
            "-c", "C:/Tirocinio/DEIMv2/configs/deimv2/deimv2_hgnetv2_atto_coco.yml",
            "--use-amp",
            "--seed", "0",
            "--device", "cuda:0",
            "--output-dir", fold_dir,
            "-u", f"dataset.train.ann={train_json}", f"dataset.val.ann={val_json}",

        ]

        print(f"\n===Training per fold {fold}: {' '.join(cmd)}===\n")
        result = subprocess.run(cmd, capture_output=True, text=True)

        log_path = os.path.join(fold_dir, "training_log.txt")
        with open(log_path, "w",  encoding="utf-8") as f:
            if result.stdout:
                f.write(result.stdout)
            else:
                f.write("No stdout returned.\n")
    
            f.write("\n\n--- STDERR ---\n")
    
            # STDERR
            if result.stderr:
                f.write(result.stderr)
            else:
                f.write("No stderr returned.\n")
            
        print(f"Log salvato in: {log_path}")

    metadata = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "random_state": 42,
        "k_folds": k,
        "config_file": "C:/Tirocinio/DEIMv2/configs/deimv2/deimv2_hgnetv2_atto_coco.yml"
    }

    save_coco_annotations(metadata, f"C:/Tirocinio/DEIMv2/folds_{k}/metadata.json")
   

if __name__ == "__main__":
    main()
