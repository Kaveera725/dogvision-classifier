import tensorflow as tf

def build_model(input_shape=(128, 128, 3)):
    """
    Builds and compiles the CNN model for Cat vs Dog classification.
    """
    
    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Preprocessing layers
        data_augmentation,
        tf.keras.layers.Rescaling(1./255, name="rescaling"),
        
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_1"),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_2"),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_3"),
        
        # Flatten and Dense Layers
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation='relu', name="dense_1"),
        tf.keras.layers.Dropout(0.5, name="dropout"),
        
        # Output Layer (Binary Classification)
        tf.keras.layers.Dense(1, activation='sigmoid', name="output")
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
