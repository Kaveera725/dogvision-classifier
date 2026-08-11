import tensorflow as tf

def build_model(input_shape=(128, 128, 3)):
    """
    Builds and compiles the CNN model for Cat vs Dog classification using Transfer Learning (MobileNetV2).
    """
    
    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ], name="data_augmentation")

    # Load pre-trained MobileNetV2 base (without the top classification layer)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model so we don't destroy its pre-trained weights during initial training
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Preprocessing layers
        data_augmentation,
        # MobileNetV2 expects input pixels to be between -1 and 1
        tf.keras.layers.Rescaling(1./127.5, offset=-1, name="rescaling"),
        
        # The pre-trained base model
        base_model,
        
        # Our custom classification head
        tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling"),
        tf.keras.layers.Dropout(0.2, name="dropout"),
        
        # Output Layer (Binary Classification: 0 for Cat, 1 for Dog)
        tf.keras.layers.Dense(1, activation='sigmoid', name="output")
    ])

    # Compile the model with a lower learning rate for transfer learning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
