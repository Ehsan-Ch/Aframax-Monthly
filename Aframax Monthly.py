"""
Time-Series Forecasting with Bidirectional LSTM
------------------------------------------------
This script provides a complete workflow for forecasting monthly values
using Bidirectional LSTM models. It includes:

- Data preprocessing (winsorization, normalization, date parsing)
- Model creation and training
- Iterative prediction with online learning
- Accuracy evaluation
- Exporting results to Excel

The dataset is not included. Replace `your_dataset.csv` with your own file.
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from scipy.stats import mstats
from sklearn.feature_selection import SelectKBest, f_regression

# ---------------------------------------------------------
# 1. Load and preprocess dataset
# ---------------------------------------------------------

DATA_PATH = "your_dataset.csv"   # Replace with your own dataset

data = pd.read_csv(DATA_PATH)

# Rename first column to Date
data = data.rename(columns={data.columns[0]: "Date"})
data = data.reset_index(drop=True)

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d", errors="coerce")
data = data.set_index("Date", drop=True)

# Ensure datetime index
data.index = pd.to_datetime(data.index)

# Shift values to avoid negatives or zeros
data = data + abs(data.min().min()) + 1

# Winsorize outliers (1st–99th percentile)
for col in data.columns:
    data[col] = mstats.winsorize(data[col], limits=[0.01, 0.01])

# Min–Max normalization
for col in data.columns:
    col_min, col_max = data[col].min(), data[col].max()
    data[col] = (data[col] - col_min) / (col_max - col_min)

# Replace zeros with tiny positive number
data[data == 0] = 1e-10

# Save normalized dataset
data.to_excel("normalized_data.xlsx")

# ---------------------------------------------------------
# 2. Model creation
# ---------------------------------------------------------

def create_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50), input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# Prepare time-series samples
def prepare_data(series, n_input):
    X, y = [], []
    for i in range(n_input, len(series)):
        X.append(series[i - n_input:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# ---------------------------------------------------------
# 3. Training setup
# ---------------------------------------------------------

targets = ["47179", "11118", "77781", "37997"]
models = {}
k_best_features = {}
n_input = 1

# ---------------------------------------------------------
# 4. Train models for each target
# ---------------------------------------------------------

for target in targets:

    # Select training window (customizable)
    if target == "47179":
        train_series = data[target].iloc[129:362].values
    else:
        train_series = data[target].iloc[:362].values

    X_train, y_train = prepare_data(train_series, n_input)

    model = create_model((n_input, 1))

    # Train sample-by-sample (online-like training)
    for i in range(len(X_train)):
        x_sample = X_train[i].reshape(1, -1)
        y_sample = y_train[i].reshape(1, -1)
        model.fit(x_sample, y_sample, epochs=100, batch_size=16, verbose=0)

    models[target] = model

    # Optional: K-best feature selection
    selector = SelectKBest(score_func=f_regression, k=5)
    if X_train.shape[1] >= 5:
        selector.fit(X_train, y_train)
        k_best_features[target] = selector.get_support(indices=True).tolist()

# ---------------------------------------------------------
# 5. Accuracy calculation
# ---------------------------------------------------------

def calculate_accuracy(actual, predicted):
    return np.abs(np.abs((actual - predicted) / actual) - 1) * 100

accuracies = {}
confidence_levels = {}

# Output DataFrame
output_cols = ["Date"]
for t in targets:
    output_cols += [f"Act.{t}", f"Pred.{t}", f"Acc.{t}"]

output_df = pd.DataFrame(columns=output_cols)

# ---------------------------------------------------------
# 6. Iterative prediction + online retraining
# ---------------------------------------------------------

for target in targets:

    accuracies[target] = []
    accurate_count = 0
    total_predictions = 0

    train_series = data[target].iloc[:362].values

    for i in range(362, len(data) - n_input):

        date_str = data.index[i + n_input].strftime("%Y-%m-%d")

        # Ensure row exists
        if date_str not in output_df["Date"].values:
            output_df.loc[len(output_df)] = [date_str] + [""] * (len(output_cols) - 1)

        row_idx = output_df.index[output_df["Date"] == date_str][0]

        # Prepare test sample
        test_sample = data[target].iloc[i:i + n_input].values.reshape(1, -1)
        prediction = models[target].predict(test_sample, verbose=0)[0][0]
        actual = data[target].iloc[i + n_input]

        # Accuracy
        acc = calculate_accuracy(actual, prediction)
        accuracies[target].append(
            f"Sample: {date_str} Actual: {actual:.4f} Predicted: {prediction:.4f} Accuracy: {acc:.2f}%"
        )

        # Save results
        output_df.at[row_idx, f"Act.{target}"] = actual
        output_df.at[row_idx, f"Pred.{target}"] = prediction
        output_df.at[row_idx, f"Acc.{target}"] = acc

        if acc >= 85:
            accurate_count += 1
        total_predictions += 1

        # Online retraining
        train_series = np.append(train_series, test_sample)
        X_train, y_train = prepare_data(train_series, n_input)
        models[target].fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # Confidence level
    confidence_levels[target] = (accurate_count / max(1, total_predictions)) * 100

# ---------------------------------------------------------
# 7. Add summary rows
# ---------------------------------------------------------

output_df.loc[len(output_df)] = ["avg.acc"] + [""] * (len(output_cols) - 1)
for target in targets:
    avg_acc = pd.to_numeric(output_df[f"Acc.{target}"], errors="coerce").mean()
    output_df.at[len(output_df) - 1, f"Acc.{target}"] = avg_acc

output_df.loc[len(output_df)] = ["conf."] + [""] * (len(output_cols) - 1)
for target in targets:
    output_df.at[len(output_df) - 1, f"Acc.{target}"] = confidence_levels[target]

# Save results
output_df.to_excel("output.xlsx", index=False)

# Print logs
for target in targets:
    print(f"\nTarget: {target}")
    for entry in accuracies[target]:
        print(entry)
    print(f"Confidence Level: {confidence_levels[target]:.2f}%")
