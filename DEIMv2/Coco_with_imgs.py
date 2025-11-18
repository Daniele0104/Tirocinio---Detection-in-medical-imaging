import json
import os
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

def binary_mask_to_coco(images_dir, masks_dir, output_json, category_name="parasite"):
    """
    Converte maschere binarie in formato COCO JSON.
    
    Args:
        images_dir: Directory con le immagini originali
        masks_dir: Directory con le maschere binarie (bianco = oggetto)
        output_json: Path del file JSON di output
        category_name: Nome della categoria degli oggetti
    """
    
    # Struttura COCO
    coco_format = {
        "info": {
            "description": "Dataset Parassiti",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": category_name,
                #"supercategory": "parasite"
            }
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    # Ottieni lista immagini (senza duplicati)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    image_files = []
    seen_names = set()
    
    for ext in image_extensions:
        for img_path in Path(images_dir).glob(f'*{ext}'):
            # Usa il nome normalizzato per evitare duplicati
            normalized_name = img_path.name.lower()
            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                image_files.append(img_path)
    
    print(f"Trovate {len(image_files)} immagini")
    
    for img_path in sorted(image_files):
        img_filename = img_path.name
        
        # Cerca la maschera corrispondente
        mask_path = Path(masks_dir) / img_filename
        
        if not mask_path.exists():
            # Prova con estensione .png se l'originale è diversa
            mask_path = Path(masks_dir) / (img_path.stem + '.png')
        
        if not mask_path.exists():
            print(f"Attenzione: maschera non trovata per {img_filename}")
            continue
        
        # Leggi immagine per ottenere dimensioni
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        
        # Aggiungi info immagine
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })
        
        # Leggi maschera
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Binarizza (bianco = 255 diventa oggetto)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Trova contorni con semplificazione
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        # Per ogni contorno (ogni istanza di parassita)
        for contour in contours:
            # Calcola area
            area = cv2.contourArea(contour)
            
            # Salta contorni troppo piccoli (probabilmente rumore)
            if area < 100:  # Aumentato per evitare rumore
                continue
            
            # Semplifica ulteriormente il contorno (riduce i punti del 95%)
            epsilon = 0.01 * cv2.arcLength(contour, True)  # 1% del perimetro (più aggressivo)
            contour_simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calcola bounding box
            x, y, w, h = cv2.boundingRect(contour_simplified)
            
            # Formato bbox COCO: [x, y, width, height] come float
            bbox = [float(x), float(y), float(w), float(h)]
            
            # Aggiungi annotazione
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "segmentation": []  # Segmentazione vuota come nel tuo formato
            })
            
            annotation_id += 1
        
        print(f"Processata: {img_filename} - Trovati {len(contours)} oggetti")
        image_id += 1
    
    # Salva JSON
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\n✓ File COCO JSON salvato in: {output_json}")
    print(f"  - Immagini: {len(coco_format['images'])}")
    print(f"  - Annotazioni: {len(coco_format['annotations'])}")


if __name__ == "__main__":
    # CONFIGURA QUESTI PATH
    IMAGES_DIR = "C:/Tirocinio/DEIMv2/dataset/ALL_parasites/img"      # Directory con le immagini originali
    MASKS_DIR = "C:/Tirocinio/DEIMv2/dataset/ALL_parasites/gt"        # Directory con le maschere binarie
    OUTPUT_JSON = "annotations_ALL_parasites.json"   # File JSON di output
    binary_mask_to_coco(IMAGES_DIR, MASKS_DIR, OUTPUT_JSON)