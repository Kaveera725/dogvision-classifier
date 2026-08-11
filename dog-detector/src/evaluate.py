import os
import shutil
import hashlib
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def check_data_integrity():
    test_cat_dir = Path("test_images/cat")
    test_dog_dir = Path("test_images/dog")
    train_cat_dir = Path("dataset/cat")
    train_dog_dir = Path("dataset/dog")
    model_path = Path("models/cat_dog_classifier.keras")
    
    # Check model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Check test directories exist
    if not test_cat_dir.exists() or not test_dog_dir.exists():
        raise FileNotFoundError("Test image directories not found.")

    test_cats = list(test_cat_dir.glob("*.*"))
    test_dogs = list(test_dog_dir.glob("*.*"))

    # Check counts
    if len(test_cats) != 100:
        raise ValueError(f"Expected 100 test cat images, found {len(test_cats)}")
    if len(test_dogs) != 100:
        raise ValueError(f"Expected 100 test dog images, found {len(test_dogs)}")

    # Check corruption
    for img_path in test_cats + test_dogs:
        if not is_valid_image(img_path):
            raise ValueError(f"Corrupted image found: {img_path}")

    # Check leakage against training/val
    train_hashes = set()
    for d in [train_cat_dir, train_dog_dir]:
        if d.exists():
            for filepath in d.glob("*.*"):
                train_hashes.add(get_file_hash(filepath))
                
    for img_path in test_cats + test_dogs:
        if get_file_hash(img_path) in train_hashes:
            raise ValueError(f"Data leakage detected! Test image {img_path} was found in the training dataset.")

    print("Data integrity checks passed.")

import sys

def evaluate(model_path):
    check_data_integrity()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model_name = Path(model_path).stem
    if model_name.endswith("_cat_dog_classifier"):
        prefix = model_name.replace("_cat_dog_classifier", "") + "_"
    elif model_name == "cat_dog_classifier":
        prefix = "baseline_"
    else:
        prefix = model_name + "_"
    
    test_dir = "test_images"
    image_size = (128, 128)
    batch_size = 32

    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load dataset with shuffle=False to match file paths with predictions
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=["cat", "dog"], # Explicitly define to force cat=0, dog=1
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = test_dataset.class_names
    if class_names != ["cat", "dog"]:
        raise ValueError(f"Class mapping mismatch! Expected ['cat', 'dog'], got {class_names}")

    # File paths for all test images
    file_paths = test_dataset.file_paths
    
    # Extract true labels
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    # Generate predictions
    print("Generating predictions...")
    preds = model.predict(test_dataset)
    pred_probs = preds.ravel()
    y_pred = (pred_probs >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    cat_precision, dog_precision = precision[0], precision[1]
    cat_recall, dog_recall = recall[0], recall[1]
    cat_f1, dog_f1 = f1[0], f1[1]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({model_name})")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{prefix}confusion_matrix.png")
    plt.close()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Analyze mistakes
    misclassified_dir = Path(f"results/{prefix}misclassified")
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    os.makedirs(misclassified_dir)

    misclassified_records = []
    
    correct_count = 0
    incorrect_count = 0

    for i in range(len(y_true)):
        actual_class = y_true[i]
        pred_class = y_pred[i]
        
        if actual_class != pred_class:
            incorrect_count += 1
            filename = Path(file_paths[i]).name
            conf = pred_probs[i] if pred_class == 1 else (1.0 - pred_probs[i])
            actual_label = class_names[actual_class]
            pred_label = class_names[pred_class]
            
            # Save image
            dest_path = misclassified_dir / filename
            shutil.copy2(file_paths[i], dest_path)
            
            misclassified_records.append({
                "filename": filename,
                "actual": actual_label,
                "predicted": pred_label,
                "confidence": conf
            })
        else:
            correct_count += 1

    if misclassified_records:
        with open(f"results/{prefix}misclassified.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "actual", "predicted", "confidence"])
            writer.writeheader()
            writer.writerows(misclassified_records)
    else:
        with open(f"results/{prefix}misclassified.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "actual", "predicted", "confidence"])
            writer.writeheader()

    # Final Summary
    print("========================================")
    print("CAT vs DOG — INDEPENDENT TEST RESULTS")
    print("========================================")
    print(f"\nTest images: {len(y_true)}\n")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}\n")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # overall metrics macro avg
    overall_precision = precision.mean() * 100
    overall_recall = recall.mean() * 100
    overall_f1 = f1.mean() * 100

    print(f"Precision: {overall_precision:.2f}%")
    print(f"Recall: {overall_recall:.2f}%")
    print(f"F1 Score: {overall_f1:.2f}%\n")

    print("CAT:")
    print(f"Precision: {cat_precision * 100:.2f}%")
    print(f"Recall: {cat_recall * 100:.2f}%")
    print(f"F1 Score: {cat_f1 * 100:.2f}%\n")

    print("DOG:")
    print(f"Precision: {dog_precision * 100:.2f}%")
    print(f"Recall: {dog_recall * 100:.2f}%")
    print(f"F1 Score: {dog_f1 * 100:.2f}%")
    print("\n========================================")
    
    print("\nComparing with Validation Performance:")
    print("Validation accuracy: 74.19%")
    print(f"Test accuracy:       {accuracy * 100:.2f}%")
    
    diff = accuracy * 100 - 74.19
    if abs(diff) <= 3:
        conclusion = "Good generalization. The test performance is very close to validation performance, suggesting the model generalizes well to unseen data without significant overfitting."
    elif diff > 3:
        conclusion = "Possible validation/test distribution difference. The model performed better on the test set, which could indicate the test set is slightly 'easier' or there's a difference in image distributions."
    elif -10 <= diff < -3:
        conclusion = "Mild overfitting. The model performs slightly worse on completely unseen data compared to the validation set."
    else:
        conclusion = "Possible significant overfitting. The test accuracy is much lower than the validation accuracy. The model failed to generalize well to this new unseen data."
        
    print(f"Conclusion: {conclusion}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/cat_dog_classifier.keras"
    evaluate(model_path)
