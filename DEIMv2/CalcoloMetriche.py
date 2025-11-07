import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os # Aggiunto per compatibilità

def analyze_evaluation(eval_file_path, json_file_path, output_dir):
    """
    Carica i risultati di una valutazione COCO (eval.pth) e salva:
    1. Un JSON con l'AP per classe.
    2. La curva Precision-Recall per classe.
    """
    
    eval_file = Path(eval_file_path)
    json_file = Path(json_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Assicura che la cartella esista

    if not eval_file.exists():
        print(f"Errore: File di valutazione non trovato in:\n{eval_file}")
        return
    if not json_file.exists():
        print(f"Errore: File JSON di validazione non trovato in:\n{json_file}")
        return

    print(f"Caricamento file di valutazione: {eval_file}")
    print(f"Caricamento file JSON: {json_file}")
    
    # --- 1. Carica i file ---
    # Ignora l'avviso di FutureWarning
    eval_results = torch.load(eval_file, map_location='cpu', weights_only=False)
    
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # --- CORREZIONE IMPORTANTE ---
    # pycocotools mappa gli ID delle categorie *ordinati* [1, 2, 3...] 
    # agli indici dell'array [0, 1, 2...]. Dobbiamo fare lo stesso.
    sorted_category_ids = sorted(categories.keys())
    class_id_map = {idx: cat_id for idx, cat_id in enumerate(sorted_category_ids)}

    # --- MODIFICA: Prepara il report JSON ---
    report = {
        'metriche_per_classe': {}
    }

    print("\n---Average Precision (AP) per Classe ---")
    
    precision_data = eval_results['precision']
    # Calcola AP (media su 10 IoU e 101 Recall)
    # Prendiamo A=0 (all areas) e M=2 (maxDets=100)
    ap_per_class = precision_data[:, :, :, 0, 2].mean(axis=(0, 1))

    for class_index, ap in enumerate(ap_per_class):
        class_id = class_id_map.get(class_index)
        if class_id is None:
             print(f"Attenzione: Indice classe {class_index} non trovato in class_id_map.")
             continue
        
        class_name = categories.get(class_id, f'Classe ID {class_id}')
        ap_value = ap.item() # Converte in float
        
        print(f"  AP [{class_name}]: {ap_value:.4f}")
        
        # --- MODIFICA: Salva il valore nel report ---
        report['AP_per_classe'][class_name] = ap_value
    
    # --- MODIFICA: Salva il Report JSON ---
    json_report_path = output_dir / "metriche_per_classe.json"
    with open(json_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report AP per classe salvato in: {json_report_path}")

    # --- 2. Salva la Curva P-R ---
    print("\nSalvataggio curva P-R...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        pr_curve_data = precision_data[0, :, :, 0, 2] # Curva a IoU=0.50
        recall_steps = np.arange(0, 1.01, 0.01)
        
        for class_index in range(pr_curve_data.shape[1]):
            class_id = class_id_map.get(class_index)
            if class_id is None:
                continue
            
            class_name = categories.get(class_id, f'Classe ID {class_id}')
            class_pr = pr_curve_data[:, class_index]
            ap_50 = class_pr.mean()
            
            if ap_50 > 0.001:
                ax.plot(recall_steps, class_pr, label=f'{class_name} (AP@.50 = {ap_50:.3f})')
        
        ax.set_title("Curve Precision-Recall (per Classe) @ IoU=0.50")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc='lower left', fontsize='small')
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.0)
        
        fig_path = output_dir / "precision_recall_curve_per_classe.png"
        fig.savefig(fig_path)
        print(f"✓ Curva P-R salvata in: {fig_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Errore durante la generazione della curva P-R: {e}")

if __name__ == "__main__":
    
    # --- Percorsi dei file ---
    # Questo ora punta al fold 2, come nel tuo script
    
    EVAL_FILE_PATH = r'folds_k3/fold_1/test_results/eval.pth'
    JSON_FILE_PATH = r'folds_k3/fold_1/val.json'
    OUTPUT_DIR_PATH = r'folds_k3/fold_1/test_results'
    
    # -------------------------
    
    analyze_evaluation(EVAL_FILE_PATH, JSON_FILE_PATH, OUTPUT_DIR_PATH)