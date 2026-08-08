import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_history():
    history_path = Path("results/training_history.json")
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} does not exist.")

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history['accuracy']) + 1)

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/accuracy.png')
    plt.close()

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss.png')
    plt.close()

    print("Graphs saved to results/accuracy.png and results/loss.png")

if __name__ == "__main__":
    plot_history()
