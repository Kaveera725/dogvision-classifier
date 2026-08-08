import os
import tensorflow as tf
from pathlib import Path

def clean():
    count = 0
    for root, _, files in os.walk("dataset"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = tf.io.read_file(file_path)
                tf.io.decode_image(img, channels=3, expand_animations=False)
            except Exception as e:
                print(f"Removing invalid TF image: {file_path}")
                os.remove(file_path)
                count += 1
    print(f"Removed {count} bad images.")

if __name__ == "__main__":
    clean()
