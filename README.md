# Aframax Monthly Forecasting

A Python research implementation for forecasting monthly shipping-related time series with Keras Bidirectional LSTM models.

The original project dataset is private and is not included. This repository contains the forecasting script and documentation.

## Implementation

The current script:

- Parses dates and prepares numeric target series.
- Applies winsorisation and min-max scaling.
- Creates a separate Bidirectional LSTM model with 50 units and a dense output for each configured target.
- Uses mean-squared-error loss and the Adam optimiser.
- Generates iterative predictions and retrains the model as the loop advances.
- Exports the processed data and prediction records.

The script currently uses a one-step input window and four target identifiers: `47179`, `11118`, `77781` and `37997`. These are project-specific settings rather than a general input contract.

## Repository contents

| File | Purpose |
| --- | --- |
| `Aframax Monthly.py` | Data preparation, model training, iterative prediction and result export |
| `README.md` | Project context, configuration and evaluation notes |

## Dependencies

The imports and exports require Python packages including pandas, NumPy, SciPy, scikit-learn, Keras, a compatible backend such as TensorFlow, and openpyxl.

```bash
python -m pip install pandas numpy scipy scikit-learn tensorflow keras openpyxl
```

Dependency versions are not pinned. Check Keras/TensorFlow compatibility and tensor shapes in your chosen environment before running a full experiment.

## Configure and run

1. Provide a CSV dataset. The first column is parsed as dates in `YYYYMMDD` format; target columns must be numeric.
2. Update `DATA_PATH` in the script.
3. Update `targets`, the hard-coded training windows and `n_input` for your data. The current loop assumes sufficient history beyond row 362.
4. Run the actual entry file:

```bash
python "Aframax Monthly.py"
```

The script writes `normalized_data.xlsx` and `output.xlsx` in the working directory. These are runtime outputs, not bundled benchmark results.

## Evaluation status

The published implementation is a research snapshot. It does not include a reproducible benchmark on the private dataset.

- Winsorisation and scaling are currently calculated before the evaluation split. A defensible forecast experiment should fit these transformations on training data only.
- The script's percentage "accuracy" is a custom formula. The field called "confidence" is the proportion of scores above a threshold, not a statistical confidence interval.
- Standard forecast errors such as MAE/RMSE, a chronological validation protocol and a simple baseline are needed before making comparative performance claims.
- Training input shapes and the indexing used during iterative retraining should be checked in a reproducible run.

## Next development steps

Make data paths, target columns and training windows configurable; pin the environment; add training-only preprocessing, standard forecast metrics and baseline comparisons; then evaluate longer input windows and multi-step forecasts.
