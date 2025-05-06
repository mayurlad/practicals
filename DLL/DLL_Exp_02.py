"""
IMDB Movie Review Sentiment Classification using Deep Neural Networks
This program implements a deep neural network to classify movie reviews as positive or negative
using the IMDB dataset.

Dataset:
- 50,000 movie reviews from IMDB
- Reviews are preprocessed and encoded as sequences of word indices
- Labels are binary (0: negative, 1: positive)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_imdb_data(num_words=10000, maxlen=200):
    """
    Load and preprocess the IMDB dataset.
    Args:
        num_words: Maximum number of words to keep based on word frequency
        maxlen: Maximum length of each review sequence
    Returns:
        Preprocessed training and testing data
    """
    print("Loading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    
    # Pad sequences to ensure uniform length
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    
    return (X_train, y_train), (X_test, y_test)

def create_model(num_words, maxlen):
    """
    Creates an improved neural network model for sentiment classification.
    The model includes:
    - Embedding layer for word vector representation
    - 1D Convolutional layer for feature extraction
    - LSTM layer for sequential pattern learning
    - Dense layers with dropout for classification
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Input and Embedding layer
        tf.keras.layers.Embedding(num_words, 128, input_length=maxlen),
        
        # 1D Convolutional layer for feature extraction
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        
        # LSTM layer for sequential pattern learning
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        
        # Dense layers with dropout
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def main():
    # Parameters
    NUM_WORDS = 10000
    MAXLEN = 200
    BATCH_SIZE = 128
    EPOCHS = 10
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_imdb_data(NUM_WORDS, MAXLEN)
    
    # Create validation split
    val_samples = 10000
    X_val = X_train[:val_samples]
    y_val = y_train[:val_samples]
    X_train = X_train[val_samples:]
    y_train = y_train[val_samples:]
    
    # Create and compile model
    model = create_model(NUM_WORDS, MAXLEN)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
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
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Make predictions on some test samples
    print("\nSample Predictions vs Actual Labels:")
    predictions = model.predict(X_test[:5])
    for pred, actual in zip(predictions, y_test[:5]):
        sentiment = "Positive" if pred > 0.5 else "Negative"
        actual_sentiment = "Positive" if actual == 1 else "Negative"
        confidence = pred if pred > 0.5 else 1 - pred
        print(f"Predicted: {sentiment} (confidence: {confidence[0]:.2f}), Actual: {actual_sentiment}")

if __name__ == "__main__":
    main()