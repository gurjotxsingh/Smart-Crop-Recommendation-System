<div align="center">

# 🌾 InteliCrop

### An Ensemble Machine Learning Model to Predict Crop

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Intelligent crop recommendation system using ensemble machine learning algorithms that combines 7 individual classifiers with Voting and Stacking strategies for maximum prediction accuracy.*

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models Used](#-models-used)
- [Results](#-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

**InteliCrop** is a Python-based mini project that predicts the most suitable crop for cultivation based on soil nutrient levels (N, P, K, pH) and weather conditions (temperature, humidity, rainfall). 

It employs an **ensemble learning** approach—combining multiple ML algorithms into a unified prediction system—to deliver significantly higher accuracy than any individual model alone.

### Problem Statement
Farmers often struggle to decide which crop to grow based on available soil and climatic conditions. Wrong crop selection leads to poor yields and economic loss.

### Solution
InteliCrop uses machine learning to analyze soil nutrients and weather data, recommending the optimal crop with confidence scores—empowering data-driven agricultural decisions.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧬 **Ensemble Learning** | Combines 7 ML algorithms with Voting & Stacking ensembles |
| 🌐 **Interactive Web App** | Beautiful Streamlit dashboard for easy crop prediction |
| 📊 **Model Comparison** | Side-by-side accuracy comparison of all 9 models |
| 🔥 **Confusion Matrix** | Detailed per-class prediction analysis |
| 📈 **Feature Importance** | Understand which factors matter most |
| 🕸️ **Radar Charts** | Visual crop nutrient/weather profiles |
| 🎯 **Confidence Scores** | Probability-based ranking of top crop recommendations |
| 📦 **Model Persistence** | Save & load trained models with joblib |
| 🧪 **Modular Codebase** | Clean, documented, production-ready structure |

---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core programming language |
| **Scikit-Learn** | ML algorithms & ensemble methods |
| **Pandas / NumPy** | Data manipulation & preprocessing |
| **Matplotlib / Seaborn** | Static visualizations |
| **Plotly** | Interactive charts |
| **Streamlit** | Web application framework |
| **Joblib** | Model serialization |

---

## 🏗 Architecture

```
Input Features                    Ensemble Layer                    Output
┌──────────────┐     ┌─────────────────────────────┐     ┌──────────────┐
│ • Nitrogen   │     │  ┌─────────────────────┐    │     │              │
│ • Phosphorus │     │  │  Random Forest       │    │     │  Predicted   │
│ • Potassium  │     │  │  Gradient Boosting   │    │     │  Crop        │
│ • Temperature│────▶│  │  Extra Trees         │────▶──▶│              │
│ • Humidity   │     │  │  Decision Tree       │    │     │  + Confidence│
│ • pH         │     │  │  KNN                 │    │     │    Scores    │
│ • Rainfall   │     │  │  SVM (RBF)           │    │     │              │
│              │     │  │  Naive Bayes          │    │     │  (22 crop   │
│              │     │  └─────────┬───────────┘    │     │   classes)   │
│              │     │            │                 │     │              │
│              │     │  ┌─────────▼───────────┐    │     │              │
│              │     │  │ Voting / Stacking    │    │     │              │
│              │     │  │ Meta-Ensemble        │    │     │              │
│              │     │  └─────────────────────┘    │     │              │
└──────────────┘     └─────────────────────────────┘     └──────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/InteliCrop.git
   cd InteliCrop
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate dataset** (if not included)
   ```bash
   python -m src.generate_dataset
   ```

---

## 💻 Usage

### Option 1: Run the Web App (Recommended)

```bash
streamlit run app.py
```

This launches an interactive dashboard where you can:
- **Predict crops** by entering soil & weather parameters
- **Compare models** with accuracy charts and confusion matrices
- **Explore the dataset** with interactive visualizations

### Option 2: Train Models via CLI

```bash
python train.py
```

This runs the full training pipeline and saves the best model.

### Option 3: Use as a Python Module

```python
from src.data_preprocessing import load_data, preprocess_data
from src.ensemble_model import run_full_pipeline, predict_crop, load_model

# Load and preprocess
df = load_data("data/Crop_recommendation.csv")
processed = preprocess_data(df)

# Train all models
results = run_full_pipeline(
    processed["X_train"], processed["X_test"],
    processed["y_train"], processed["y_test"],
    label_encoder=processed["label_encoder"],
)

# Predict with saved model
bundle = load_model()
result = predict_crop(bundle, N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=202)
print(f"Recommended Crop: {result['predicted_crop']}")
```

---

## 🤖 Models Used

### Individual Models

| # | Model | Description |
|---|-------|-------------|
| 1 | **Random Forest** | Bagging ensemble of 200 decision trees |
| 2 | **Gradient Boosting** | Sequential boosting with 150 estimators |
| 3 | **Extra Trees** | Extremely randomized trees (200 estimators) |
| 4 | **Decision Tree** | Single tree with max depth 12 |
| 5 | **K-Nearest Neighbors** | Distance-weighted with k=5 |
| 6 | **SVM (RBF Kernel)** | Support Vector Machine with radial basis function |
| 7 | **Naive Bayes** | Gaussian Naive Bayes classifier |

### Ensemble Methods

| Method | Strategy | Description |
|--------|----------|-------------|
| **Voting Ensemble** | Soft Voting | Averages predicted probabilities from all 7 models |
| **Stacking Ensemble** | Stacking | Uses Logistic Regression as meta-learner over 7 base models |

---

## 📊 Results

The ensemble approach consistently outperforms individual models:

| Model | Accuracy |
|-------|----------|
| Stacking Ensemble | ~98-99% |
| Voting Ensemble | ~97-98% |
| Random Forest | ~97-98% |
| Gradient Boosting | ~96-97% |
| Extra Trees | ~97-98% |
| SVM (RBF) | ~95-97% |
| KNN | ~95-97% |
| Decision Tree | ~93-95% |
| Naive Bayes | ~90-93% |

> *Exact values depend on the dataset and random seed. Run `train.py` to see your results.*

---

## 📁 Dataset

### Features

| Feature | Description | Unit |
|---------|-------------|------|
| N | Ratio of Nitrogen content in soil | ppm |
| P | Ratio of Phosphorus content in soil | ppm |
| K | Ratio of Potassium content in soil | ppm |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | pH value of soil | - |
| rainfall | Rainfall | mm |

### Target
- **label**: Name of the crop (22 classes)

### Crops Supported
Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 📂 Project Structure

```
InteliCrop/
├── app.py                     # Streamlit web application
├── train.py                   # CLI training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── generate_dataset.py    # Synthetic dataset generator
│   ├── data_preprocessing.py  # Data loading & preprocessing
│   ├── ensemble_model.py      # ML models & ensemble logic
│   └── visualizations.py      # Plotting utilities
│
├── data/                      # Dataset directory
│   └── Crop_recommendation.csv
│
├── models/                    # Saved trained models
│   └── intelicrop_model.pkl
│
├── notebooks/                 # Jupyter notebooks (EDA)
│
└── static/                    # Static assets
```

---

## 🖼 Screenshots

*Run the app with `streamlit run app.py` to see the beautiful interactive dashboard!*

- **Home Page** — Overview of the system with quick stats
- <img width="1918" height="995" alt="image" src="https://github.com/user-attachments/assets/fddc20c7-02f7-4944-a791-0964b05a7706" />

- **Predict Crop** — Interactive sliders for input parameters with confidence charts
- <img width="1916" height="991" alt="image" src="https://github.com/user-attachments/assets/db01cb46-0e7a-46bf-b67c-7cf47f747519" />

- **Model Analysis** — Comparative accuracy charts and confusion matrices
- 
- **Dataset Explorer** — Interactive histograms, radar charts, and correlation heatmaps

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---




<div align="center">

**Made with ❤️ for Smart Agriculture**

⭐ Star this repo if you found it useful!

</div>
