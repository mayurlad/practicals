"""
Boston Housing Price Prediction using Deep Neural Networks
This program implements a deep neural network to predict housing prices using the Boston Housing dataset.

Dataset Features:
- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxides concentration (parts per 10 million)
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population

Target Variable:
- MEDV: Median value of owner-occupied homes in $1000s
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_boston_housing_data():
    """
    Load and preprocess the Boston Housing dataset from its original source.
    Returns:
        X: normalized feature matrix
        y: normalized target values
        scaler_y: the target scaler for inverse transformation
    """
    # Load the Boston Housing dataset from original source
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # Normalize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(data)
    y = scaler_y.fit_transform(target.reshape(-1, 1))
    
    return X, y, scaler_y

# Create an improved neural network model with regularization and batch normalization
def create_model():
    """
    Creates an improved neural network model for housing price prediction.
    The model includes:
    - Batch normalization for better training stability
    - Dropout layers for regularization
    - Multiple hidden layers with regularization
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(13,)),
        tf.keras.layers.BatchNormalization(),
        
        # First hidden layer
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Second hidden layer
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Third hidden layer
        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer
        tf.keras.layers.Dense(1)
    ])
    
    return model

# Load and preprocess data
X, y, scaler_y = load_boston_housing_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create TensorFlow datasets with proper batching
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Create and compile model
model = create_model()

# Learning rate schedule for better convergence
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

# Update the optimizer with fixed learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Display model summary
print("\nModel Architecture:")
model.summary()

# Callbacks for training
callbacks = [
    # Early stopping to prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when training plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
]

# Training
print("\nTraining the model...")
history = model.fit(
    train_dataset,
    epochs=200,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Evaluate on test set
test_results = model.evaluate(test_dataset, verbose=0)
print(f"\nTest Mean Absolute Error (normalized): {test_results[1]:.4f}")

# Make predictions on test set
test_predictions = model.predict(X_test)

# Convert predictions back to original scale
test_predictions_original = scaler_y.inverse_transform(test_predictions)
test_actual_original = scaler_y.inverse_transform(y_test)

# Calculate MAE in original scale (thousands of dollars)
mae_original = np.mean(np.abs(test_predictions_original - test_actual_original))
print(f"\nTest Mean Absolute Error: ${mae_original:.2f}k")

# Show sample predictions in original scale
print("\nSample Predictions vs Actual Values (in $1000s):")
for pred, actual in zip(test_predictions_original[:5], test_actual_original[:5]):
    print(f"Predicted: ${pred[0]:.2f}k, Actual: ${actual[0]:.2f}k")

# Additional model performance metrics
mse = np.mean((test_predictions_original - test_actual_original) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - (np.sum((test_actual_original - test_predictions_original) ** 2) / 
          np.sum((test_actual_original - np.mean(test_actual_original)) ** 2))

print("\nAdditional Performance Metrics:")
print(f"Root Mean Square Error: ${rmse:.2f}k")
print(f"R-squared Score: {r2:.4f}")