import json
import os

# --- CONFIGURAZIONE ---

# 1. Inserisci il percorso del tuo file di annotazioni originale (quello con 6 classi)
#    (Usiamo 'trainval.json' come input per 'fold.py')
ORIGINAL_JSON_PATH = "dataset/annotations/instances_test.json"

# 2. Inserisci il nome del file che vuoi creare
FILTERED_JSON_PATH = "dataset/annotations/instances_test_only_parasites.json"

# 3. Metti i nomi di TUTTE le classi che vuoi RAGGRUPPARE
CLASSES_TO_MERGE = [
    "gametocyte",
    "ring",
    "schizont",
    "trophozoite"
]

# 4. Definisci la tua nuova classe singola
NEW_CLASS_NAME = "parasite"
NEW_CLASS_ID = 1  
# --- FINE CONFIGURAZIONE ---


def filter_and_merge_coco_json():
    print(f"Caricamento di {ORIGINAL_JSON_PATH}...")
    with open(ORIGINAL_JSON_PATH, 'r') as f:
        coco_data = json.load(f)

    print("Creazione mappatura categorie...")
    
    # 1. Crea la mappa degli ID
    #    Mappa tutti gli ID delle classi da unire al NUOVO ID singolo 
    old_id_to_new_id = {}
    original_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for old_id, name in original_categories.items():
        if name in CLASSES_TO_MERGE:
            old_id_to_new_id[old_id] = NEW_CLASS_ID
    
    # 2. Crea la nuova lista delle categorie (che conterrà una sola voce)
    new_categories = [
        {"id": NEW_CLASS_ID, "name": NEW_CLASS_NAME, "supercategory": NEW_CLASS_NAME}
    ]
    
    print(f"Mappate {len(CLASSES_TO_MERGE)} classi nella nuova classe '{NEW_CLASS_NAME}' (ID: {NEW_CLASS_ID})")

    # 3. Filtra le annotazioni
    print("Filtraggio e unione delle annotazioni...")
    new_annotations = []
    annotations_kept = 0
    annotations_discarded = 0

    # Tieni traccia degli ID delle immagini per il filtraggio
    image_ids_with_annotations = set()

    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        
        # Se l'ID è uno di quelli che vogliamo unire...
        if old_cat_id in old_id_to_new_id:
            new_ann_entry = ann.copy()
            # ...assegnagli il NUOVO ID singolo (0)
            new_ann_entry['category_id'] = old_id_to_new_id[old_cat_id]
            new_annotations.append(new_ann_entry)
            
            image_ids_with_annotations.add(ann['image_id']) # Salva l'ID dell'immagine
            annotations_kept += 1
        else:
            # Scarta "red blood cell", "difficult", ecc.
            annotations_discarded += 1 

    print(f"Annotazioni mantenute: {annotations_kept}, Scartate: {annotations_discarded}")

    # 4. Filtra le immagini (rimuove quelle senza annotazioni)
    print("Filtraggio delle immagini...")
    new_images = [img for img in coco_data['images'] if img['id'] in image_ids_with_annotations]
    
    print(f"Immagini mantenute: {len(new_images)} su {len(coco_data['images'])}")

    # 5. Crea il nuovo file JSON
    new_coco_data = {
        "info": coco_data.get('info', {}),
        "licenses": coco_data.get('licenses', []),
        "categories": new_categories, # Usa la nuova lista di categorie (singola)
        "images": new_images,
        "annotations": new_annotations
    }

    # 6. Salva il file
    print(f"Salvataggio in corso su {FILTERED_JSON_PATH}...")
    with open(FILTERED_JSON_PATH, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    print("Operazione completata!")


if __name__ == "__main__":
    filter_and_merge_coco_json()