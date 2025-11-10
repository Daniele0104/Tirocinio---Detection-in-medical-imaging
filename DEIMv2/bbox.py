import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from engine.core import YAMLConfig

# ===== CONFIG =====
config_path = "configs/deimv2/deimv2_hgnetv2_atto_coco.yml"  # <-- percorso al tuo YAML
checkpoint_path = "folds_k3/fold_0/best_stg2.pth"            # <-- percorso al modello
num_images = 5                                               # numero immagini da mostrare
score_threshold = 0.5                                        # soglia di confidenza

# ===== CARICA MODELLO =====
cfg = YAMLConfig(config_path)

# costruisci il modello
model = cfg.model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# carica i pesi
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    print("✅ Modello caricato correttamente!")

except RuntimeError as e:
    print(f"❌ ERRORE: Mismatch tra i pesi di 'checkpoint_path' e l'architettura di 'config_path'.")
    print("Assicurati che 'config_path' e 'checkpoint_path' siano compatibili.")
    print(f"Dettagli: {e}")
    sys.exit(1)


# costruisci postprocessor (se la repo lo fornisce)
try:
    postprocessor = cfg.postprocessor
except Exception:
    postprocessor = None
    print("⚠️ Postprocessor non trovato, verranno mostrate solo le box raw.")

# costruisci dataset di validazione
dataset = cfg.val_dataloader.dataset

if dataset is None:
    print(f"❌ ERRORE: Impossibile caricare il dataset di validazione.")
    print(f"Controlla che 'val_dataloader:' -> 'dataset:' sia corretto in:")
    print(f"{config_path} (e nei suoi file __include__)")
    sys.exit(1)

print(f"✅ Dataset validazione caricato: {len(dataset)} immagini\n")


# ===== VISUALIZZA ALCUNE IMMAGINI =====
for idx in range(num_images):
    img, target = dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)  # aggiunge batch dimension

    with torch.no_grad():
        output = model(img_tensor)

    # post-processing (se disponibile)
    if postprocessor is not None:
        
        orig_size_tensor = target['orig_size'].unsqueeze(0).long().to(device)
        results = postprocessor(output, orig_size_tensor)
        
        res = results[0]
        boxes = res['boxes'].cpu().numpy()
        scores = res['scores'].cpu().numpy()
        labels = res['labels'].cpu().numpy()
    else:
        # fallback: box grezze (solo se non serve postprocess)
        print("⚠️ Sto usando il fallback per le box (postprocessor non trovato).")
        boxes = output['pred_boxes'][0].cpu().numpy()
        scores = output['pred_logits'][0].softmax(-1).max(-1)[0].cpu().numpy()
        labels = output['pred_logits'][0].softmax(-1).argmax(-1).cpu().numpy()

    # ===== BLOCCO CORRETTO PER CARICARE L'IMMAGINE =====
    try:
        # 1. Ottieni l'ID dell'immagine dal target
        image_id = target['image_id'].item()  # .item() converte un tensore 0D in un numero
        
        # 2. Usa l'API COCO del dataset per ottenere le info dell'immagine
        img_info = dataset.coco.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        
        # 3. Costruisci il percorso completo
        # (dataset.img_folder è definito nel file .yml incluso)
        img_path = os.path.join(dataset.img_folder, file_name)

    except Exception as e:
        print(f"⚠️ Errore nel recuperare il percorso per l'indice {idx}: {e}, salto.")
        continue
    
    # 4. Controlla se il file esiste e caricalo
    if not os.path.exists(img_path):
        print(f"⚠️ Immagine non trovata per l'indice {idx} al percorso: {img_path}, salto.")
        print(f"(Controlla che 'img_folder' in 'coco_detection.yml' sia corretto)")
        continue
        
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # ===================================================

    # disegna bounding box
    found_box = False
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        found_box = True
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_cv, f"{label}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    if not found_box:
        print(f"Nessuna box trovata per l'immagine {idx+1} con soglia > {score_threshold}")

    # mostra con matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img_cv)
    plt.axis('off')
    plt.title(f"Example {idx + 1}")
    plt.show()