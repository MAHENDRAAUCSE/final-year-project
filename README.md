# Final Year Project: Smart STP Prediction System

This project predicts wastewater treatment plant (STP) water-quality parameters using deep learning.

It includes:
- A Python ML pipeline for training CNN-LSTM-Attention models.
- Future forecasting scripts for next-day predictions.
- A FastAPI backend for inference.
- A Flutter app for uploading CSV data and viewing predictions.

## Objectives

- Forecast key STP parameters such as `BOD`, `COD`, `PH`, and `DO`.
- Compare actual vs predicted values with evaluation metrics and plots.
- Provide a simple API and mobile interface for practical use.

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- FastAPI + Uvicorn
- Flutter (Dart)

## Repository Structure

```text
project_final/
|-- data/
|   `-- Rajahmundry_STP_Daily_Synthetic_2020_2023.csv
|-- models/
|   |-- cnn_lstm_attention_model.h5
|   |-- cod_cnn_lstm_attention_model.h5
|   |-- do_cnn_lstm_attention_model.h5
|   `-- ph_cnn_lstm_attention_model.h5
|-- plots/
|-- reports/
|-- src/
|   |-- cnn_lstm_attention.py
|   `-- preprocessing.py
|-- smart_stp_predictor/
|   |-- backend/
|   |   |-- api.py
|   |   |-- predictor.py
|   |   `-- model_loader.py
|   `-- mobile_app/
|       `-- lib/
|-- generate_data.py
|-- main.py
|-- train_simple.py
|-- future_prediction.py
|-- requirements.txt
`-- README.md
```

## Dataset

- Main dataset: `data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv`
- Columns include:
	- `Date`
	- `BOD (mg/L)`
	- `COD (mg/L)`
	- `TSS (mg/L)`
	- `TN (mg/L)`
	- `TP (mg/L)`
	- `PH`
	- `DO (mg/L)`

To regenerate synthetic data:

```bash
python generate_data.py
```

## Setup (Python)

From project root (`e:\project_final`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model Training

### 1) Simple training (default BOD)

```bash
python train_simple.py
```

### 2) Target-specific training (recommended)

`main.py` supports configurable targets and data path.

Examples:

```bash
python main.py --target "BOD (mg/L)"
python main.py --target "COD (mg/L)"
python main.py --target "PH"
python main.py --target "DO (mg/L)"
```

Outputs are saved in:
- `models/` (`*_cnn_lstm_attention_model.h5`)
- `reports/` (`*_metrics.json`, `*_detailed_report.txt`)
- `plots/` (`*_training_history.png`, predictions, residuals, scatter, metrics)

## Future Prediction

Use `future_prediction.py` to forecast future values from a trained model.

Example:

```bash
python future_prediction.py --target "BOD (mg/L)" --days 7 --time_steps 30
```

The script writes an Excel file (for example `Future_BOD_Predictions.xlsx`) containing predictions and summary statistics.

## Backend API (FastAPI)

Location: `smart_stp_predictor/backend`

Install backend dependencies:

```bash
cd smart_stp_predictor/backend
pip install -r requirements.txt
```

Run server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoint

- `POST /predict`
- Input: multipart file upload with key `file` (CSV)
- Output: JSON with key `future_predictions` (7 predicted values)

## Flutter Mobile App

Location: `smart_stp_predictor/mobile_app`

Install dependencies:

```bash
cd smart_stp_predictor/mobile_app
flutter pub get
```

Run app:

```bash
flutter run
```

Notes:
- For Android emulator, API base URL is configured as `http://10.0.2.2:8000`.
- For Web, API base URL is configured as `http://localhost:8000`.
- Ensure FastAPI backend is running before uploading CSV in the app.

## Evaluation Metrics

The pipeline reports:
- RMSE
- MAE
- R2 Score
- MAPE (%)

## Troubleshooting

- If model loading fails with `GetItem` layer error, the project includes a compatibility `GetItem` class in `future_prediction.py` and `smart_stp_predictor/backend/model_loader.py`.
- If output Excel file is locked/open, close it and rerun prediction.
- If API upload fails, verify backend is running on port `8000` and CORS/network access is valid.

## GitHub

Repository: `https://github.com/MAHENDRAAUCSE/final-year-project`

README added and intended to be maintained as project modules evolve.
