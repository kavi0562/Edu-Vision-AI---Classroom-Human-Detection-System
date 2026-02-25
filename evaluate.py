from ultralytics import YOLO
import os
import glob
import math
from tqdm import tqdm

def print_evaluation_metrics():
    print("Loading model for evaluation...")
    
    # Try loading the newly trained model, otherwise fallback
    model_path = "classroom_ai/max_accuracy_run/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Falling back to default lightweight yolov8n.pt for faster CPU inference...")
        model_path = "yolov8n.pt"

    model = YOLO(model_path)

    print("\n[1/2] Running Validation on Dataset to compute Detection Scores...")
    # Run validation (make sure dataset.yaml is correctly configured)
    metrics = model.val(data="dataset.yaml", split="val", verbose=False)

    print("\n" + "#"*80)
    print(" " * 25 + "COMPULSORY METRICS REPORT" + " " * 25)
    print("#"*80)
    
    # ---------------- 1. DETECTION SCORE (mAP) ----------------
    # Extract mAP@0.5:0.95
    map_50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
    map_50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
    
    print("\n🧠 Detection Score (50%)")
    print(f"   Mean Average Precision (mAP@0.5:0.95) : {map_50_95:.4f}")
    print(f"   Mean Average Precision (mAP@0.5)      : {map_50:.4f}\n")
    
    # ---------------- 2. COUNTING SCORE (MAE) ----------------
    print("\n[2/2] Calculating Counting Score (MAE) across validation images...")
    
    val_images_dir = "dataset/valid/images"
    val_labels_dir = "dataset/valid/labels"
    
    if not os.path.exists(val_images_dir):
        print(f"⚠️ Warning: Validation images folder not found at {val_images_dir}")
        print("   Cannot calculate MAE. Please ensure dataset structure is correct.")
    else:
        # Filter only valid image extensions to prevent YOLO crashing on .npy files
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        all_files = glob.glob(os.path.join(val_images_dir, "*.*"))
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]
        
        total_error = 0
        total_images = 0
        
        # Add tqdm progress bar here
        for img_path in tqdm(image_files, desc="Counting Persons", unit="image"):
            # Get Ground Truth Count from txt label file
            filename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(val_labels_dir, filename + ".txt")
            
            true_count = 0
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    true_count = sum(1 for line in f if line.strip())
            
            # Get Predicted Count from Model
            results = model(img_path, conf=0.5, iou=0.45, verbose=False)
            pred_count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) == 0: # Check if class is person (0)
                    pred_count += 1
            
            # Calculate Absolute Error
            error = abs(pred_count - true_count)
            total_error += error
            total_images += 1
            
        if total_images > 0:
            mae = total_error / total_images
            print("\n🔢 Counting Score (50%)")
            print(f"   Mean Absolute Error (MAE)            : {mae:.4f}")
            print(f"   (Calculated across {total_images} validation images)")
        else:
            print("   No images found to calculate MAE.")

    # ---------------- 3. CLASSIFICATION REPORT ----------------
    print("\n" + "-"*80)
    print(" " * 25 + "Classification Details" + " " * 25)
    print("-"*80)
    print(f"{'class':>15} {'precision':>12} {'recall':>10} {'f1-score':>10} {'support':>10}\n")

    p = metrics.results_dict.get('metrics/precision(B)', 0.0)
    r = metrics.results_dict.get('metrics/recall(B)', 0.0)
    f1 = 2 * (p * r) / (p + r + 1e-16)
    cm = metrics.confusion_matrix.matrix
    
    if cm.shape[0] > 0:
        support = int(cm[0, :].sum())
        tp = int(cm[0, 0])
        fn = int(cm[0, -1]) 
        fp = int(cm[-1, 0]) 
        
        print(f"{'person (0)':>15} {p:>12.2f} {r:>10.2f} {f1:>10.2f} {support:>10}\n")
        print(f"{'accuracy':>15} {'':>12} {'':>10} {f1:>10.2f} {support:>10}")
        print(f"{'macro avg':>15} {p:>12.2f} {r:>10.2f} {f1:>10.2f} {support:>10}")
        print(f"{'weighted avg':>15} {p:>12.2f} {r:>10.2f} {f1:>10.2f} {support:>10}\n")
        
        print("Validation confusion matrix:")
        print(f"[[{tp:>5} {fn:>5}]")
        print(f" [ {fp:>4}     0]]") 
    else:
        print("No validation data found in confusion matrix.")

    print("#"*80)

if __name__ == "__main__":
    print_evaluation_metrics()