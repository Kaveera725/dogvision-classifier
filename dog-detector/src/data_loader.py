import tensorflow as tf
from pathlib import Path

def load_data(dataset_dir="dataset"):
    """
    Loads images from the dataset directory, resizes them to 128x128,
    splits them into 80% training and 20% validation, and normalizes
    pixels to 0-1.
    """
    image_size = (128, 128)
    batch_size = 32
    seed = 42

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

    # Check for expected class names
    if set(class_names) != {'cat', 'dog'}:
        print(f"Warning: Expected classes ['cat', 'dog'], but found {class_names}")

    # Normalize pixels from 0-255 to 0-1
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    validation_dataset = validation_dataset.map(
        lambda x, y: (normalization_layer(x), y), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optimize pipeline performance
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Print summary
    print(f"\nClasses: {class_names}")
    
    # tf.keras.utils.image_dataset_from_directory adds file_paths attribute
    if hasattr(train_dataset, 'file_paths'):
        print(f"Training images: {len(train_dataset.file_paths)}")
        print(f"Validation images: {len(validation_dataset.file_paths)}")
    else:
        print(f"Training batches: {len(train_dataset)}")
        print(f"Validation batches: {len(validation_dataset)}")
        
    return train_dataset, validation_dataset, class_names

if __name__ == "__main__":
    train_ds, val_ds, classes = load_data()
