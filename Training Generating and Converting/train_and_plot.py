import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

try:
    from tensorflow.keras.utils import plot_model

    PLOT_MODEL_AVAILABLE = True
except ImportError:
    PLOT_MODEL_AVAILABLE = False


# Step 1: Define and Train the Model
def train_model():
    """
    Trains a simple neural network and saves the training history.
    """
    # Create a simple model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(100,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate some example data
    X_train = np.random.rand(1000, 100)
    y_train = np.random.randint(0, 10, 1000)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10)

    # Save the training history to a file
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    # Save a visualization of the model's architecture (optional)
    if PLOT_MODEL_AVAILABLE:
        try:
            plot_model(model, to_file="model_architecture.png", show_shapes=True)
            print("Model architecture saved to 'model_architecture.png'.")
        except Exception as e:
            print(f"Could not save model architecture visualization: {e}")
    else:
        print("`plot_model` is not available. Install `pydot` and `graphviz` to enable model visualization.")

    print("Training complete. History saved to 'training_history.json'.")


# Step 2: Plot Training Metrics
def plot_training_metrics(history_path, save_path="training_metrics.png"):
    """
    Plots the training metrics such as accuracy and loss from a saved Keras History object.
    """
    # Load the history file
    with open(history_path, "r") as f:
        history = json.load(f)

    # Extract data
    accuracy = history.get('accuracy', [])
    val_accuracy = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    epochs = range(1, len(accuracy) + 1)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training metrics plot saved to '{save_path}'.")


# Step 3: Main Execution
if __name__ == "__main__":
    # Train the model and save the history
    train_model()

    # Plot the training metrics
    plot_training_metrics("training_history.json")
