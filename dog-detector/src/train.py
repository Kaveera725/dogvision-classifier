import os
import json
import tensorflow as tf
from pathlib import Path

# Import our custom modules
from data_loader import load_data
from model import build_model

def train():
    print("Initializing training pipeline...")

    # 1. Load data
    train_dataset, validation_dataset, class_names = load_data(dataset_dir="dataset")

    # 2. Verify dataset
    if set(class_names) != {'cat', 'dog'}:
        raise ValueError(f"Invalid classes. Expected ['cat', 'dog'], got {class_names}")

    if not train_dataset or not validation_dataset:
        raise ValueError("Training or validation dataset is empty.")

    # 3. Create model
    model = build_model()
    model.summary()

    # 4. Verify model output is appropriate for binary classification
    output_layer = model.layers[-1]
    if output_layer.units != 1 or output_layer.activation.__name__ != 'sigmoid':
         raise ValueError(f"Model output layer must have 1 unit and sigmoid activation. Found: {output_layer.units}, {output_layer.activation.__name__}")

    # 5. Setup directories
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    # 6. Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_path = "models/mobilenetv2_cat_dog_classifier.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 7. Train model
    print("\nStarting training for up to 20 epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # 8. Save training history
    history_dict = history.history
    # Convert numpy types to python floats for JSON serialization
    history_dict_serializable = {k: [float(val) for val in v] for k, v in history_dict.items()}
    
    history_path = "results/training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict_serializable, f, indent=4)
        
    # 9. Output final results
    print("\nTraining completed.")
    
    best_val_loss = min(history_dict['val_loss'])
    best_val_acc = max(history_dict['val_accuracy'])
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"\nModel saved to:\n{checkpoint_path}")
    print(f"Training history saved to:\n{history_path}")

if __name__ == "__main__":
    train()
