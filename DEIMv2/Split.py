import json
import os
import random
import shutil
from collections import defaultdict

# --- CONFIGURAZIONE ---
INPUT_COCO_JSON = 'C:/Tirocinio/DEIMv2/dataset_prova/annotations_Falciparum.json'
SOURCE_IMAGE_DIR = "C:/Tirocinio/DEIMv2/dataset/ALL_parasites/img"
OUTPUT_DIR = 'C:/Tirocinio/DEIMv2/dataset_prova'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42
# ------------------------

def create_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'test'), exist_ok=True)

def process_split(image_list, split_name, categories, all_annotations_map):
    output_coco = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_output_dir = os.path.join(OUTPUT_DIR, 'images', split_name)

    for img_info in image_list:
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        output_coco['images'].append(img_info)
        
        if image_id in all_annotations_map:
            output_coco['annotations'].extend(all_annotations_map[image_id])
            
        src_path = os.path.join(SOURCE_IMAGE_DIR, file_name)
        dst_path = os.path.join(image_output_dir, file_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            pass # Ignora file mancanti

    json_path = os.path.join(OUTPUT_DIR, 'annotations', f'instances_{split_name}.json')
    
    with open(json_path, 'w') as f:
        json.dump(output_coco, f, indent=2)

def split_coco_dataset():
    if not os.path.exists(SOURCE_IMAGE_DIR):
        raise FileNotFoundError(
            f"La cartella sorgente '{SOURCE_IMAGE_DIR}' non esiste. "
            "Modifica la variabile 'SOURCE_IMAGE_DIR'."
        )

    create_dirs()

    with open(INPUT_COCO_JSON, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    annotations_map = defaultdict(list)
    for ann in annotations:
        annotations_map[ann['image_id']].append(ann)

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    total_count = len(images)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count : train_count + val_count]
    test_images = images[train_count + val_count :]

    process_split(train_images, 'train', categories, annotations_map)
    process_split(val_images, 'val', categories, annotations_map)
    process_split(test_images, 'test', categories, annotations_map)

if __name__ == "__main__":
    split_coco_dataset()