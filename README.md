# Aframax-Monthly
Forecasting monthly values ​​of shipping companies.

---

# 📈 Time‑Series Forecasting with Bidirectional LSTM  
*A modular framework for predicting monthly values in shipping and logistics*

This repository contains a complete workflow for building **Bidirectional LSTM–based time‑series forecasting models**.  
Although originally developed for predicting **monthly operational values of shipping companies**, the code is fully generalizable and can be applied to any univariate time‑series dataset.

> **Note:** The dataset used in development is private and therefore not included in this repository.  
> You can replace it with your own CSV file following the structure described below.

---

## 🚀 Features

- **Full preprocessing pipeline**  
  - Date parsing and indexing  
  - Outlier handling via Winsorization  
  - Min‑Max normalization  
  - Zero‑value correction for numerical stability  

- **Neural network architecture**  
  - Bidirectional LSTM with a single-step forecasting setup  
  - Customizable input window (`n_input`)  
  - Sample‑by‑sample training option for incremental learning  

- **Dynamic / online learning**  
  - After each prediction, the new data point is appended to the training set  
  - The model retrains itself continuously to adapt to new patterns  

- **Feature selection (optional)**  
  - K‑Best regression scoring for exploratory analysis  

- **Evaluation metrics**  
  - Custom accuracy formula  
  - Average accuracy per target  
  - Confidence level based on threshold performance  

- **Exportable results**  
  - Predictions, actual values, and accuracy stored in a structured Excel file  

---

## 📂 Project Structure

```
.
├── README.md
├── main.py                # Core forecasting script
├── output.xlsx            # Generated predictions (created at runtime)
└── normalized_data.xlsx   # Preprocessed dataset (created at runtime)
```

---

## 📊 How the Model Works

### 1. **Data Preparation**
The script loads a CSV file where the first column represents dates and the remaining columns represent numeric time‑series features.  
It performs:

- Date conversion  
- Index resetting  
- Outlier trimming  
- Normalization  
- Zero‑value correction  

This ensures the data is clean and stable for neural network training.

---

### 2. **Model Architecture**
Each target variable is modeled independently using:

- A **Bidirectional LSTM** layer with 50 units  
- A **Dense** output layer  
- `mean_squared_error` loss  
- `adam` optimizer  

This architecture captures both forward and backward temporal dependencies.

---

### 3. **Training Strategy**
The model uses a **1‑step lag** (`n_input = 1`) to predict the next value.

Training occurs in two phases:

#### **Initial Training**
A historical window is selected for each target, and the model is trained sample‑by‑sample.

#### **Iterative Forecasting + Online Retraining**
For each new time step:

1. Predict the next value  
2. Calculate accuracy  
3. Append the new sample to the training set  
4. Retrain the model on the expanded dataset  

This creates an **adaptive forecasting system** that evolves with the data.

---

### 4. **Evaluation**
The script computes:

- Accuracy for each prediction  
- Average accuracy across all predictions  
- A “confidence level” based on how many predictions exceed an accuracy threshold (default: 85%)  

All results are exported to `output.xlsx`.

---

## 📁 Input Data Format

Your CSV file should follow this structure:

| Date       | Feature1 | Feature2 | ... |
|------------|----------|----------|-----|
| 19910101   | 123      | 456      | ... |
| 19910201   | 130      | 470      | ... |
| ...        | ...      | ...      | ... |

- The **first column must contain dates** in `YYYYMMDD` format.  
- All other columns must be numeric.  
- You may include as many features as you want; the script selects specific targets internally.

---

## 🛠 Requirements

Install dependencies using:

```
pip install pandas numpy keras scipy scikit-learn openpyxl
```

---

## ▶️ Running the Script

Update the dataset path in the code:

```python
data = pd.read_csv('path/to/your/data.csv')
```

Then run:

```
python main.py
```

The script will generate:

- `normalized_data.xlsx`  
- `output.xlsx`  
- Console logs with detailed accuracy information  

---

## 🔒 About the Dataset

The original dataset used to develop this project is **private** and cannot be shared.  
However, the code is fully reusable with any time‑series dataset that follows the required structure.

---

## 📌 Future Improvements

- Multi‑step forecasting  
- Hyperparameter tuning  
- Support for multivariate LSTM inputs  
- Visualization of predictions and error metrics  

---

If you'd like, I can also help you generate a **LICENSE**, **contribution guidelines**, or a **clean folder structure** for publishing this on GitHub.
