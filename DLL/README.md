# Deep Learning Laboratory

## List of Programs

Each experiment is provided in both Python script (.py) and Jupyter Notebook (.ipynb) formats.

### 1. Boston Housing Price Prediction using Deep Neural Network

- Files:
  - [Python Script](DLL_Exp_01.py)
  - [Jupyter Notebook](DLL_Exp_01.ipynb) (Recommended)
- Dataset: Boston House Price Dataset
  - Download: Built into tensorflow.keras.datasets
  - [Dataset Description](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
- Model: Deep Neural Network
- Type: Regression Problem

### 2. IMDB Movie Review Sentiment Classification

- Files:
  - [Python Script](DLL_Exp_02.py)
  - [Jupyter Notebook](DLL_Exp_02.ipynb) (Recommended)
- Dataset: IMDB Movie Reviews Dataset (50,000 reviews)
  - Download: Built into tensorflow.keras.datasets
  - [Dataset Description](https://ai.stanford.edu/~amaas/data/sentiment/)
- Model: Deep Neural Network with Embedding, Conv1D, and LSTM layers
- Type: Binary Classification Problem

### 3. Fashion MNIST Classification using CNN

- Files:
  - [Python Script](DLL_Exp_03.py)
  - [Jupyter Notebook](DLL_Exp_03.ipynb) (Recommended)
- Dataset: Fashion MNIST Dataset
  - Download: Built into tensorflow.keras.datasets
  - [Dataset Description](https://github.com/zalandoresearch/fashion-mnist)
- Model: Convolutional Neural Network
- Type: Multi-class Classification Problem

### 4. Google Stock Price Prediction using RNN

- Files:
  - [Python Script](DLL_Exp_04.py)
  - [Jupyter Notebook](DLL_Exp_04.ipynb) (Recommended)
- Dataset: Google (GOOGL) Stock Price History
  - Download: Automatic using yfinance
  - Data Range: Last 5 years of daily prices
- Model: Stacked LSTM (Long Short-Term Memory)
- Type: Time Series Prediction
- Features:
  - Real-time data fetching
  - Sliding window sequence generation
  - Future price prediction
  - Interactive visualizations

## Dataset Information

All datasets used in these implementations are conveniently available through TensorFlow's built-in datasets module (`tensorflow.keras.datasets`). They will be automatically downloaded when running the respective scripts for the first time.

To manually download and explore these datasets:
```python
# Boston Housing Dataset
from tensorflow.keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# IMDB Reviews Dataset
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# Fashion MNIST Dataset
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```

## Project Structure
```
DLL/
├── DLL_Exp_01.py            # Housing price prediction
├── DLL_Exp_01.ipynb         # Housing price prediction (Notebook)
├── DLL_Exp_02.py            # Sentiment analysis
├── DLL_Exp_02.ipynb         # Sentiment analysis (Notebook)
├── DLL_Exp_03.py            # Fashion item classification
├── DLL_Exp_03.ipynb         # Fashion item classification (Notebook)
├── DLL_Exp_04.py            # Stock price prediction
├── DLL_Exp_04.ipynb         # Stock price prediction (Notebook)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Getting Started

1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter:
```bash
jupyter notebook
```

4. Open any of the .ipynb files to run the experiments interactively

## Dependencies
All required packages are listed in `requirements.txt`:
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter
- IPython
