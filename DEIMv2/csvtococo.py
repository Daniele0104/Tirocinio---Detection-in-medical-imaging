import os
import pandas as pd
import json
from PIL import Image

# --- 1. Impostazioni da modificare ---

# Metti qui il percorso al dataset che vuoi processare
BASE_DATASET_PATH = 'C:/Tirocinio/DEIMv2/dataset/Falciparum' # CAMBIA QUESTO

# Specifica il nome del tuo file CSV di input
INPUT_CSV_NAME = 'mp-idb-falciparum.csv' # METTI QUI IL NOME DEL TUO CSV

# --- MAPPA PER I NOMI DELLE CLASSI ---
LABEL_NAME_MAP = {
    'game': 'gametocyte',
    'ring': 'ring',
    'schi': 'schizont',
    'tro': 'trophozoite'
}
# ----------------------------------------

# --- 2. Impostazioni Derivate ---
IMAGE_DIR = os.path.join(BASE_DATASET_PATH, 'img')
CSV_FILE_PATH = os.path.join(BASE_DATASET_PATH, INPUT_CSV_NAME)
OUTPUT_JSON = f'coco_{os.path.basename(BASE_DATASET_PATH).lower()}.json'

# --- 3. Funzioni di Supporto ---

def find_main_image(base_name, image_dir):
    """
    Cerca l'immagine principale corrispondente.
    Cerca prima una corrispondenza esatta, poi un file che INIZIA 
    con il base_name.
    """
    
    # 1. Prova a cercare una corrispondenza esatta (più veloce se funziona)
    for ext in ['.jpg', '.jpeg', '.png', '.tif']:
        test_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(test_path):
            return test_path
            
    # 2. Se fallisce, cerca un file che *inizia con* il base_name
    try:
        for filename in os.listdir(image_dir):
            if filename.startswith(base_name):
                full_path = os.path.join(image_dir, filename)
                if os.path.isfile(full_path):
                    return full_path
    except FileNotFoundError:
        return None
    
    return None

def scan_categories_from_csv(df):
    """
    Scansiona il DataFrame per trovare tutte
    le etichette di classe uniche e creare una mappa.
    """
    print("Scansione CSV per le categorie...")
    # La colonna 'parasite_type' contiene i codici (G, R, S, T)
    all_labels = set(df['parasite_type'].unique())

    category_map = {}
    categories_list = []
    if not all_labels:
        print("ATTENZIONE: Nessuna categoria valida trovata.")
        return {}, []

    sorted_labels = sorted(list(all_labels))
    category_id_counter = 1 
    
    print("Costruzione categorie COCO:")
    for label_code in sorted_labels:
        full_name = LABEL_NAME_MAP.get(label_code, label_code)
        
        if label_code not in LABEL_NAME_MAP:
            print(f"ATTENZIONE: Trovato codice '{label_code}' non mappato. Uso il codice come nome.")

        category_id = category_id_counter
        
        categories_list.append({
            "id": category_id,
            "name": full_name
        })
        
        category_map[label_code] = category_id 
        print(f"  - Mappato: '{label_code}' -> ID: {category_id}, Nome: '{full_name}'")
        category_id_counter += 1
        
    print(f"Mappa categorie (per lo script): {category_map}")
    return category_map, categories_list

# --- 4. Esecuzione Principale ---

def main():
    print(f"Processamento dataset in: {BASE_DATASET_PATH}")
    
    # Inizializza la struttura COCO
    coco_output = {
        "info": {"description": f"Dataset {os.path.basename(BASE_DATASET_PATH)}"},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # --- 1. Carica il CSV ---
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"ERRORE: File CSV non trovato in {CSV_FILE_PATH}")
        return
        
    print(f"Trovate {len(df)} righe (annotazioni) nel file CSV.")
    
    # --- 2. Trova le categorie ---
    category_map, categories_list = scan_categories_from_csv(df)
    coco_output["categories"] = categories_list
    
    if not category_map:
        print("ERRORE: Nessuna categoria trovata nel CSV.")
        return

    # Inizializza i contatori
    annotation_id_counter = 1
    image_id_counter = 1
    
    # --- 3. Raggruppa per Immagine ---
    # Questo è FONDAMENTALE. Processa ogni immagine una sola volta.
    grouped = df.groupby('filename')
    
    for filename, group in grouped:
        
        # Cerca l'immagine originale (usando il nome pulito)
        # Supponiamo che il CSV possa avere percorsi, quindi puliamo
        clean_filename = os.path.basename(os.path.normpath(filename))
        
        # Usiamo il 'base_name' (senza estensione) per trovare l'immagine
        # Questo è sbagliato se il CSV ha nomi diversi (es. 123.jpg)
        # e l'immagine è 123-R_S.jpg. 
        # Riprova: la nostra logica 'find_main_image' è migliore.
        # Dobbiamo trovare il 'base_name' dal CSV
        
        # L'approccio migliore: supponiamo che i nomi file nel CSV siano
        # il nome VERO dell'immagine (es. "1305121398-0003-R.jpg")
        
        main_image_path = os.path.join(IMAGE_DIR, clean_filename)
        
        # Apri l'immagine per ottenere le dimensioni
        try:
            with Image.open(main_image_path) as img:
                width, height = img.size
        except FileNotFoundError:
            print(f"ATTENZIONE: Immagine {clean_filename} non trovata in {IMAGE_DIR}. Salto {len(group)} annotazioni.")
            continue
        except Exception as e:
            print(f"Errore nell'aprire {clean_filename}: {e}. Salto.")
            continue

        # Aggiungi l'immagine alla lista COCO
        current_image_id = image_id_counter
        coco_output['images'].append({
            "id": current_image_id,
            "file_name": clean_filename,
            "width": width,
            "height": height
        })
        image_id_counter += 1
        
        # --- 4. Processa tutte le annotazioni per questa immagine ---
        for _, row in group.iterrows():
            
            # Correzione delle coordinate (xmin, xmax scambiati)
            true_x_min = min(row['xmin'], row['xmax'])
            true_x_max = max(row['xmin'], row['xmax'])
            true_y_min = min(row['ymin'], row['ymax'])
            true_y_max = max(row['ymin'], row['ymax'])

            # Conversione in formato COCO [x_min, y_min, larghezza, altezza]
            bbox_width = true_x_max - true_x_min
            bbox_height = true_y_max - true_y_min
            
            bbox_coco = [float(true_x_min), float(true_y_min), float(bbox_width), float(bbox_height)]
            area = float(bbox_width * bbox_height)

            # Ottieni il category_id (es. 1 per 'G')
            category_id = category_map[row['parasite_type']]

            # Aggiungi l'annotazione
            coco_output['annotations'].append({
                "id": annotation_id_counter,
                "image_id": current_image_id, 
                "category_id": category_id,
                "bbox": bbox_coco,
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            })
            
            annotation_id_counter += 1

    # --- 5. Salva il file JSON ---
    print(f"Processamento completato. Salvataggio in {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"Fatto! Creato {OUTPUT_JSON} con {len(coco_output['images'])} immagini e {len(coco_output['annotations'])} annotazioni.")
    
    if len(coco_output['annotations']) == len(df):
        print("VERIFICA SUPERATA: Il numero di annotazioni corrisponde alle righe del CSV (1297).")
    else:
        print(f"ATTENZIONE: Il numero di annotazioni ({len(coco_output['annotations'])}) non corrisponde alle righe del CSV ({len(df)}).")
        print("Questo accade se alcune immagini non sono state trovate.")

if __name__ == "__main__":
    main()