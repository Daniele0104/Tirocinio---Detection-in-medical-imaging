import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_evaluation(eval_file_path, json_file_path, output_dir):
    """
    Carica i risultati di una valutazione COCO (eval.pth) e salva:
    1. Un JSON con l'AP per classe.
    2. La curva Precision-Recall per classe.
    """
    
    eval_file = Path(eval_file_path)
    json_file = Path(json_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) 

    
    print(f"Caricamento file di valutazione: {eval_file}")
    print(f"Caricamento file JSON: {json_file}")
    
    # Carica i file 
    # Ignora l'avviso di FutureWarning
    eval_results = torch.load(eval_file, map_location='cpu', weights_only=False)
    
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    sorted_category_ids = sorted(categories.keys())
    class_id_map = {idx: cat_id for idx, cat_id in enumerate(sorted_category_ids)}

    # report JSON
    report = {
        'AP_per_classe': {}
    }

    print("\n---Average Precision (AP) per Classe ---")
    
    precision_data = eval_results['precision']
    ap_per_class = precision_data[:, :, :, 0, 2].mean(axis=(0, 1))

    for class_index, ap in enumerate(ap_per_class):
        class_id = class_id_map.get(class_index)
        
        class_name = categories.get(class_id, f'Classe ID {class_id}')
        ap_value = ap.item()
        
        print(f"  AP [{class_name}]: {ap_value:.4f}")
 
        report['AP_per_classe'][class_name] = ap_value
    
    # Salva report
    json_report_path = output_dir / "metriche_per_classe.json"
    with open(json_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print("\nReport AP per classe salvato")

    #Salva la Curva P-R
    print("\nSalvataggio curva P-R...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        pr_curve_data = precision_data[0, :, :, 0, 2]
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
        print("Curva P-R salvata")
        plt.close(fig)

    except Exception as e:
        print(f"Errore durante la generazione della curva P-R: {e}")

if __name__ == "__main__":
    
    # Percorso file
    
    EVAL_FILE_PATH = r'folds_k3/fold_0/test_results/eval.pth'
    JSON_FILE_PATH = r'folds_k3/fold_0/val.json'
    OUTPUT_DIR_PATH = r'folds_k3/fold_0/test_results'
    
    analyze_evaluation(EVAL_FILE_PATH, JSON_FILE_PATH, OUTPUT_DIR_PATH)