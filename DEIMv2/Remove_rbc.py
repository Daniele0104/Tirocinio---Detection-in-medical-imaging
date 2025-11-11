import json

# --- CONFIGURAZIONE ---

# 1. Inserisci il percorso del tuo file di annotazioni originale
ORIGINAL_JSON_PATH = "dataset\\annotations\\instances_test.json"

# 2. Inserisci il nome del file che vuoi creare
FILTERED_JSON_PATH = "dataset\\annotations\\instances_test_no_rbc.json"

# 3. Inserisci i nomi ESATTI delle classi che vuoi MANTENERE
#    (tutte le altre verranno scartate)
CLASSES_TO_KEEP = [
    "gametocyte",
    "ring",
    "schizont",
    "trophozoite"
]
# --- FINE CONFIGURAZIONE ---


def filter_coco_json():
    print(f"Caricamento di {ORIGINAL_JSON_PATH}...")
    with open(ORIGINAL_JSON_PATH, 'r') as f:
        coco_data = json.load(f)

    print("Filtraggio delle categorie...")
    
    # 1. Filtra le categorie e crea la mappa degli ID
    new_categories = []
    old_id_to_new_id = {}
    new_id_counter = 1 # I nuovi ID COCO devono essere sequenziali da 1

    for category in coco_data['categories']:
        if category['name'] in CLASSES_TO_KEEP:
            # Mappa il vecchio ID al nuovo ID sequenziale
            old_id = category['id']
            old_id_to_new_id[old_id] = new_id_counter
            
            # Crea la nuova voce di categoria
            new_cat_entry = category.copy()
            new_cat_entry['id'] = new_id_counter
            new_categories.append(new_cat_entry)
            
            new_id_counter += 1

    print(f"Mantenute {len(new_categories)} classi su {len(coco_data['categories'])}.")
    
    if not new_categories:
        print("ERRORE: Nessuna classe mantenuta. Controlla l'elenco 'CLASSES_TO_KEEP' e i nomi nel JSON.")
        return

    # 2. Filtra le annotazioni usando la mappa degli ID
    print("Filtraggio delle annotazioni (potrebbe richiedere tempo)...")
    new_annotations = []
    annotations_kept = 0
    annotations_discarded = 0

    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        
        # Se l'ID della categoria è tra quelli che vogliamo tenere...
        if old_cat_id in old_id_to_new_id:
            # ...crea una nuova annotazione e aggiorna il suo ID
            new_ann_entry = ann.copy()
            new_ann_entry['category_id'] = old_id_to_new_id[old_cat_id]
            new_annotations.append(new_ann_entry)
            annotations_kept += 1
        else:
            annotations_discarded += 1

    print(f"Annotazioni mantenute: {annotations_kept}, Scartate: {annotations_discarded}")

    # 3. (Opzionale ma consigliato) Filtra le immagini
    # Mantiene solo le immagini che hanno ancora almeno un'annotazione
    print("Filtraggio delle immagini...")
    image_ids_with_annotations = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in coco_data['images'] if img['id'] in image_ids_with_annotations]
    
    print(f"Immagini mantenute: {len(new_images)} su {len(coco_data['images'])}")


    # 4. Crea il nuovo file JSON
    new_coco_data = {
        "info": coco_data.get('info', {}),
        "licenses": coco_data.get('licenses', []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations
    }

    # 5. Salva il file
    print(f"Salvataggio in corso su {FILTERED_JSON_PATH}...")
    with open(FILTERED_JSON_PATH, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    print("Operazione completata!")


if __name__ == "__main__":
    filter_coco_json()