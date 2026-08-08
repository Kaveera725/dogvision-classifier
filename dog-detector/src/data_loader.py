import tensorflow as tf
import os
from pathlib import Path

def load_data(dataset_dir="dataset"):
    """
    Loads images from the dataset directory, resizes them to 128x128,
    and splits them into 80% training and 20% validation.
    No augmentation or normalization is applied here because they are 
    handled directly within the model architecture.
    """
    image_size = (128, 128)
    batch_size = 32
    seed = 42
    
    cat_dir = Path(dataset_dir) / "cat"
    dog_dir = Path(dataset_dir) / "dog"
    
    # Check the dataset before training
    if not cat_dir.exists() or not dog_dir.exists():
        raise ValueError(f"Dataset directories missing. Ensure {cat_dir} and {dog_dir} exist.")
        
    # Check if they contain images
    cat_images = list(cat_dir.glob("*.*"))
    dog_images = list(dog_dir.glob("*.*"))
    
    if len(cat_images) == 0 or len(dog_images) == 0:
        raise ValueError("One or both dataset directories are empty.")
    
    print(f"Loading datasets from '{dataset_dir}'...")
    
    # Create training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    # Create validation dataset
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_dataset.class_names

    if set(class_names) != {'cat', 'dog'}:
        raise ValueError(f"Expected classes ['cat', 'dog'], but found {class_names}")

    # Optimize pipeline performance
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Print summary
    print(f"\nClasses: {class_names}")
    print(f"Training batches: {len(train_dataset)}")
    print(f"Validation batches: {len(validation_dataset)}")
        
    return train_dataset, validation_dataset, class_names

if __name__ == "__main__":
    train_ds, val_ds, classes = load_data()
