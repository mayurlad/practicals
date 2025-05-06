"""
Fashion MNIST Classification using Convolutional Neural Networks
This program implements a CNN to classify fashion items into 10 categories using the Fashion MNIST dataset.

Dataset:
- 70,000 grayscale images of fashion items (60,000 training, 10,000 testing)
- 10 categories of clothing
- Image size: 28x28 pixels
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_fashion_mnist():
    """
    Load and preprocess the Fashion MNIST dataset
    Returns:
        Preprocessed training and testing data
    """
    print("Loading Fashion MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to include channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (X_train, y_train), (X_test, y_test)

def create_cnn_model():
    """
    Creates a CNN model for Fashion MNIST classification
    The architecture includes:
    - Multiple convolutional layers with increasing filters
    - MaxPooling layers for spatial dimension reduction
    - Batch normalization for training stability
    - Dropout for regularization
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """
    Plot the training and validation accuracy/loss curves
    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_sample_predictions(model, X_test, y_test, num_samples=10):
    """
    Plot sample predictions with actual and predicted labels
    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels
        num_samples: Number of samples to display
    """
    # Class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Get predictions
    predictions = model.predict(X_test[:num_samples])
    
    # Create figure
    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        true_label = class_names[np.argmax(y_test[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    
    # Create validation split
    val_samples = 5000
    X_val = X_train[:val_samples]
    y_val = y_train[:val_samples]
    X_train = X_train[val_samples:]
    y_train = y_train[val_samples:]
    
    # Create and compile model
    model = create_cnn_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Training parameters
    EPOCHS = 20
    BATCH_SIZE = 64
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_sample_predictions(model, X_test, y_test)

if __name__ == "__main__":
    main()