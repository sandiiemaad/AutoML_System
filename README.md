# AutoML Assistant
### Intelligent No-Code Machine Learning Platform

AutoML Assistant is a full-stack machine learning platform that enables users to build, train, evaluate, and export machine learning models without writing a single line of code.

The system automatically performs data preprocessing, feature engineering, model selection, hyperparameter optimization, imbalance handling, evaluation, and model serialization through an intuitive web interface.

---

## Features

### Dataset Management
- Upload CSV datasets
- Upload Excel datasets (.xlsx)
- Automatic dataset preview
- Column type detection

### Automated Data Preprocessing
The system automatically performs:

- Missing value handling
- Median imputation for numerical features
- Most frequent imputation for categorical features
- High-cardinality feature removal
- ID column detection and removal
- Constant feature removal
- High-missing-value feature removal
- Standard scaling
- One-Hot Encoding
- Ordinal Encoding
- Automatic preprocessing pipeline generation

### Classification
Supported models:

- Logistic Regression
- Random Forest Classifier

Automatic:

- Hyperparameter tuning
- Cross-validation
- Best model selection

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Regression
Supported models:

- Linear Regression
- Random Forest Regressor

Evaluation Metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

### Clustering
Supported models:

- K-Means Clustering
- Hierarchical Clustering

Automatic:

- Optimal cluster number detection
- Silhouette score evaluation

### Class Imbalance Handling
For classification tasks:

- Automatic imbalance detection
- SMOTE oversampling
- Balanced training dataset generation

### Model Export
- Save trained models as `.pkl`
- Export complete preprocessing + model pipeline
- Ready for deployment

---

## System Architecture

```text
Dataset Upload
      │
      ▼
Data Validation
      │
      ▼
Automated Preprocessing
      │
      ▼
Task Selection
 ┌───────────────┬───────────────┬───────────────┐
 │Classification│ Regression    │ Clustering    │
 └───────────────┴───────────────┴───────────────┘
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Export (.pkl)
```

---

## Technologies Used

### Backend

- Python
- Flask
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Joblib

### Frontend

- HTML5
- CSS3
- JavaScript (Vanilla JS)

### Machine Learning

- Logistic Regression
- Random Forest
- Linear Regression
- K-Means
- Agglomerative Clustering

---

## Project Structure

```text
AutoML-System/
│
├── app.py
├── index.html
├── script.js
│
├── models/
│
├── datasets/
│
├── screenshots/
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/AutoML-System.git
cd AutoML-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Packages

```bash
pip install flask
pip install flask-cors
pip install pandas
pip install numpy
pip install scikit-learn
pip install imbalanced-learn
pip install openpyxl
pip install joblib
```

---

## Running the Application

### Start Backend

```bash
python app.py
```

Backend runs on:

```text
http://127.0.0.1:5000
```

### Open Frontend

Open:

```text
index.html
```

in your browser.

---

## Workflow

### Step 1
Upload dataset

### Step 2
Select task:

- Classification
- Regression
- Clustering

### Step 3
(Optional)

Choose ordinal columns

### Step 4
Run preprocessing

### Step 5
Train models

### Step 6
Review results

### Step 7
Download trained model

---

## Example Results

### Classification

| Metric | Score |
|----------|----------|
| Accuracy | 96.5% |
| Precision | 95.7% |
| Recall | 96.2% |
| F1 Score | 95.9% |

### Regression

| Metric | Score |
|----------|----------|
| R² | 0.91 |
| MAE | 1.82 |
| MSE | 4.31 |

### Clustering

| Metric | Score |
|----------|----------|
| Silhouette Score | 0.78 |
| Optimal Clusters | 4 |

---

## Future Enhancements

- XGBoost integration
- LightGBM integration
- CatBoost integration
- Feature importance visualization
- Automated EDA dashboard
- Explainable AI (SHAP)
- Deep Learning support
- Cloud deployment
- Multi-user authentication
- Model monitoring

---

## Academic Contribution

This project demonstrates the integration of:

- Data Preprocessing Automation
- Machine Learning Pipeline Engineering
- Hyperparameter Optimization
- AutoML Concepts
- Model Evaluation
- Web-Based ML Systems


---

## Authors

Developed by:

- Sandy Emad
- Rawan Rushdy
- Karin Zaki
- Youssef Ahmed
  

Faculty of Computers and Artificial Intelligence
Information Systems Department
Cairo University

---
