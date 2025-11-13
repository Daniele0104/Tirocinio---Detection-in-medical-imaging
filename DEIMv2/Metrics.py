import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_evaluation(eval_file_path, json_file_path, output_dir):
    """
    Carica i risultati di una valutazione COCO (eval.pth) e salva:
    1. Un JSON con l'AP (mAP) e F1 per classe (IoU 0.50:0.95).
    2. La curva Precision-Recall Media (averaged over IoU 0.50:0.95).
    """
    
    eval_file = Path(eval_file_path)
    json_file = Path(json_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) 
    
    print(f"Caricamento file di valutazione: {eval_file}")
    print(f"Caricamento file JSON: {json_file}")
    
    try:
        eval_results = torch.load(eval_file, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"File non trovato: {eval_file}")
        return
    
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    sorted_category_ids = sorted(categories.keys())
    class_id_map = {idx: cat_id for idx, cat_id in enumerate(sorted_category_ids)}

    # report JSON
    report = {
        'AP_per_classe': {},
        'F1_per_classe_0.50_0.95': {}
    }

    print("\n--- Metriche per Classe (IoU 0.50:0.95) ---")
    
    # Shape: [IoU, Recall, Class, Area, MaxDets]
    precision_data = eval_results['precision']
    
    # Calcolo AP standard COCO (Media su IoU e Recall)
    ap_per_class = precision_data[:, :, :, 0, 2].mean(axis=(0, 1))
    recall_steps = np.arange(0, 1.01, 0.01)

    for class_index, ap in enumerate(ap_per_class):
        class_id = class_id_map.get(class_index)
        class_name = categories.get(class_id, f'Classe ID {class_id}')
        ap_value = ap.item()
        
        # --- CALCOLO F1 SCORE (IoU 0.50:0.95) ---
        class_precisions_all_iou = precision_data[:, :, class_index, 0, 2]
        f1_matrix = 2 * (class_precisions_all_iou * recall_steps) / (class_precisions_all_iou + recall_steps + 1e-16)
        best_f1_per_iou = np.max(f1_matrix, axis=1)
        f1_mean_value = np.mean(best_f1_per_iou)
        # ----------------------------------------

        print(f"  {class_name:<20} | AP: {ap_value:.4f} | F1 (0.50:0.95): {f1_mean_value:.4f}")
 
        report['AP_per_classe'][class_name] = ap_value
        report['F1_per_classe_0.50_0.95'][class_name] = float(f1_mean_value)
    
    # Salva report
    json_report_path = output_dir / "metriche_per_classe.json"
    with open(json_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print("\nReport metriche salvato")

    # --- CURVA P-R MEDIA (IoU 0.50:0.95) ---
    print("\nSalvataggio curva P-R (Media 0.50:0.95)...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # PRENDO LA MEDIA SULL'ASSE 0 (IoU)
        # precision_data[:, :, :, 0, 2] ha shape [10, 101, NumClasses]
        # .mean(axis=0) collassa le 10 IoU in una media -> shape [101, NumClasses]
        pr_curve_data_mean = precision_data[:, :, :, 0, 2].mean(axis=0)
        
        for class_index in range(pr_curve_data_mean.shape[1]):
            class_id = class_id_map.get(class_index)
            if class_id is None:
                continue
            
            class_name = categories.get(class_id, f'Classe ID {class_id}')
            
            # Curva media per questa classe
            class_pr_mean = pr_curve_data_mean[:, class_index]
            
            # L'AP qui è matematicamente identico all'AP calcolato sopra (la media delle medie è commutativa)
            ap_val = class_pr_mean.mean()
            
            if ap_val > 0.001:
                ax.plot(recall_steps, class_pr_mean, label=f'{class_name} (mAP = {ap_val:.3f})')
        
        ax.set_title("Mean Precision-Recall Curve (IoU=0.50:0.95)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision (Averaged over IoU)")
        ax.legend(loc='lower left', fontsize='small')
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.0)
        
        fig_path = output_dir / "mean_precision_recall_curve_050_095.png"
        fig.savefig(fig_path)
        print("Curva P-R salvata")
        plt.close(fig)

    except Exception as e:
        print(f"Errore durante la generazione della curva P-R: {e}")

if __name__ == "__main__":
    print("=== ANALISI NO RBC ===")
    for k in range(3):
        fold_name = f'fold_{k}'
        base_path = Path(f'folds_k3/{fold_name}/test_results')
        EVAL_FILE_PATH = base_path / 'eval.pth'
        JSON_FILE_PATH = rf'dataset/annotations/instances_test_no_rbc.json'
        
        if EVAL_FILE_PATH.exists():
            analyze_evaluation(EVAL_FILE_PATH, JSON_FILE_PATH, base_path)

    print("\n=== ANALISI CAT PARASITES ===")
    for k in range(3):
        fold_name = f'fold_{k}'
        base_path = Path(f'folds_onlyP_k3/{fold_name}/test_results')
        EVAL_FILE_PATH = base_path / 'eval.pth'
        JSON_FILE_PATH = rf'dataset/annotations/instances_test_only_parasites.json'
        
        if EVAL_FILE_PATH.exists():
            analyze_evaluation(EVAL_FILE_PATH, JSON_FILE_PATH, base_path)