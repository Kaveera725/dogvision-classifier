import sys
import tensorflow as tf
import numpy as np
import matplotlib
import PIL
import sklearn

def main():
    print("--- Environment Verification ---")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Pillow version: {PIL.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print("--------------------------------")
    print("\nEnvironment setup is successful! All packages imported correctly.")

    
if __name__ == "__main__":
    main()
