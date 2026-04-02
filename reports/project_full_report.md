# Smart STP Prediction System - Comprehensive Project Report

## 1. Project Title

Smart STP Prediction System for Wastewater Quality Forecasting Using a CNN-LSTM-Attention Model

## 2. Project Overview

This project focuses on predicting wastewater treatment plant (STP) water-quality parameters using deep learning. The system is designed to learn patterns from daily wastewater time-series data and forecast future values of important quality indicators. In addition to the machine learning pipeline, the project also includes a FastAPI backend for inference and a Flutter mobile application for CSV upload and prediction display.

The implemented pipeline in this repository follows the workflow below:

Data preprocessing -> CNN feature extraction -> LSTM temporal learning -> Attention-based focus -> Prediction -> Metric evaluation -> Visualization -> Deployment through API and mobile app

## 3. Model Used

The main model used in this project is a hybrid deep learning model built with three major components:

### CNN (Convolutional Neural Network)

- Extracts local temporal patterns from the input time window.
- Implemented using 1D convolution layers.
- Helps capture short-term feature relationships in the wastewater sequence data.

### LSTM (Long Short-Term Memory)

- Learns long-term dependencies in time-series data.
- Processes sequential behavior across daily measurements.
- Useful for forecasting wastewater trends over time.

### Attention Mechanism

- Highlights the most important time steps from the LSTM output.
- Improves the model's ability to focus on informative parts of the sequence.
- Supports better sequence-to-one prediction performance.

### Final Model

The final architecture used is:

CNN-LSTM-Attention

### Implemented Architecture Details

- Input shape: `(30, 7)` for a 30-day history window and 7 numeric input features.
- CNN branch:
  - Conv1D with 64 filters
  - BatchNormalization
  - Conv1D with 32 filters
  - MaxPooling1D
  - Dropout
- LSTM branch:
  - LSTM with 64 units
  - Dropout
  - LSTM with 32 units
  - Dropout
- Attention branch:
  - Self-attention over LSTM outputs
  - GlobalAveragePooling1D
- Fusion:
  - Concatenation of CNN and attention outputs
- Dense head:
  - Dense 64 -> Dense 32 -> Dense 16 -> Dense 1

### Model Size

- Model name: `CNN_LSTM_Attention`
- Total parameters: 45,729
- Trainable parameters: 45,473
- Non-trainable parameters: 256

## 4. Programming Language and Frameworks Used

### Core Language

- Python is used as the main language for data preprocessing, model training, evaluation, and future prediction.

### Deep Learning Framework

- TensorFlow / Keras is used to define, train, save, and load the neural network model.

### Backend Framework

- FastAPI is used to expose a prediction API endpoint.
- Uvicorn is used to run the backend server.

### Mobile Application Framework

- Flutter is used to build the mobile and web interface.
- Dart is used for the mobile application code.

## 5. Data Handling and Processing Libraries Used

The following libraries are used in the repository:

- Pandas: data loading, tabular processing, CSV handling, result export.
- NumPy: numerical operations, array transformation, sequence preparation.
- scikit-learn:
  - MinMaxScaler for feature scaling.
  - Evaluation helpers through metric functions.

## 6. Visualization Tools Used

The project uses:

- Matplotlib for training curves, prediction plots, scatter plots, residual plots, and metric charts.
- Seaborn is listed in the project requirements for enhanced visualization support.

The training pipeline generates plots in the `plots/` directory such as:

- training history
- actual vs predicted comparison
- residual analysis
- metrics summary
- actual vs predicted scatter plot

## 7. Development Environment Used

The PDF summary mentions Jupyter Notebook and Google Colab for experimentation. In the current repository, the final implementation is organized as Python scripts rather than notebooks. The effective development stack represented in this workspace is:

- Python scripts for training and inference
- TensorFlow / Keras for model development
- FastAPI for service deployment
- Flutter for the front-end client
- Standard local development environment with saved reports, models, and plots

## 8. Dataset Used

### Dataset Name

`data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv`

### Dataset Scope

- Daily wastewater data from 2020 to 2023
- Total rows: 1,461
- Date range: 2020-01-01 to 2023-12-31

### Parameters Included

The dataset contains these fields:

- Date
- BOD (mg/L)
- COD (mg/L)
- TSS (mg/L)
- TN (mg/L)
- TP (mg/L)
- PH
- DO (mg/L)

### Data Characteristics

The dataset is synthetic but structured to imitate realistic wastewater behavior. The generation script introduces:

- seasonal patterns
- trend components
- noise
- autocorrelation between consecutive days
- realistic physical bounds on each parameter

### Approximate Statistical Summary

- Mean BOD: 165.31 mg/L
- Mean COD: 322.51 mg/L
- Mean TSS: 208.79 mg/L
- Mean TN: 39.85 mg/L
- Mean TP: 4.99 mg/L
- Mean PH: 7.20
- Mean DO: 4.75 mg/L

## 9. Preprocessing and Data Preparation Used

The project implements the following preprocessing pipeline:

### Data Loading

- CSV data is loaded using Pandas.
- The `Date` column is removed from the training frame when building the model inputs.

### Train-Test Split

- The dataset is split into training and testing sets using an 80:20 ratio.

### Feature Scaling

- MinMax scaling is applied using `MinMaxScaler`.
- The scaler is fit on the training set and then applied to the test set.

### Sequence Generation

- Supervised time-series sequences are created using a sliding window.
- Window size: 30 days.
- Each sample uses 30 past days to predict the next value of the selected target variable.

### Inverse Transformation

- Predictions are inverse-scaled back to the original units for reporting and comparison.

### Synthetic Data Preparation

The `generate_data.py` script creates the dataset with:

- seasonal sinusoidal variation
- noise injection
- trend progression
- autocorrelation smoothing
- clipping and bounds enforcement for valid wastewater values

## 10. Training Setup Used

The implemented training configuration in the main training pipeline is:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean Squared Error (MSE)
- Training metric: MAE
- Epochs: 100
- Batch size: 32
- Early stopping patience: 15
- Reduce learning rate on plateau patience: 8
- Minimum learning rate: 1e-7
- Train-test split: 80% training, 20% testing
- Window size: 30 time steps

### Targets Trained in the Project

The repository includes trained models and reports for:

- BOD
- COD
- PH
- DO

### Saved Models

Examples already present in the project:

- `models/cnn_lstm_attention_model.h5`
- `models/cod_cnn_lstm_attention_model.h5`
- `models/ph_cnn_lstm_attention_model.h5`
- `models/do_cnn_lstm_attention_model.h5`
- `smart_stp_predictor/backend/models/bod_model.h5`

## 11. Evaluation and Analysis Used

The project uses the following evaluation methods:

### Metrics

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R2 Score
- MAPE: Mean Absolute Percentage Error

### Visual Analysis

- Actual vs predicted trend plots
- Training loss and MAE curves
- Scatter plot of actual vs predicted values
- Residual plot
- Metric summary bar charts

### Existing Performance Results in This Repository

#### BOD

- RMSE: 14.4834
- MAE: 11.8191
- R2 Score: 0.8289
- MAPE: 7.3658%

#### COD

- RMSE: 28.3680
- MAE: 22.9936
- R2 Score: 0.8343
- MAPE: 7.5561%

#### PH

- RMSE: 0.0725
- MAE: 0.0583
- R2 Score: 0.5148
- MAPE: 0.8159%

#### DO

- RMSE: 0.5045
- MAE: 0.4189
- R2 Score: 0.5356
- MAPE: 8.5569%

### Interpretation of Results

- BOD and COD show the strongest predictive performance among the saved reports.
- PH and DO are predicted with lower R2 scores, indicating these targets are harder to model with the current setup.
- The system is still able to produce stable forecasts for multiple important wastewater indicators.

## 12. Future Prediction Module

The repository includes a future forecasting module through `future_prediction.py`.

### Supported Functionality

- Loads a trained `.h5` model
- Selects a target column
- Uses the last 30 time steps of data
- Predicts future values for a specified number of days
- Saves prediction outputs to an Excel file

### Output Contents

The generated Excel output contains:

- predicted dates
- predicted target values in original scale
- model output in scaled form
- summary statistics such as mean, minimum, and maximum predicted values

## 13. Backend API Component

The project includes a deployed inference backend inside `smart_stp_predictor/backend`.

### API Features

- Built with FastAPI
- CORS enabled for client access
- Endpoint: `POST /predict`
- Accepts CSV file upload
- Returns 7 future predictions in JSON format

### Backend Prediction Logic

- Reads uploaded CSV data
- Selects numeric columns
- Applies MinMax scaling
- Uses a 30-day window
- Loads the trained backend model
- Predicts the next 7 days
- Returns predicted values in original scale

## 14. Mobile Application Component

The repository also includes a Flutter mobile application in `smart_stp_predictor/mobile_app`.

### App Purpose

- Upload a CSV data file
- Send the file to the FastAPI backend
- Receive predicted values
- Display results to the user

### Client Integration Details

- Android API base URL: `http://10.0.2.2:8000`
- Web API base URL: `http://localhost:8000`

This makes the project a complete end-to-end smart prediction system rather than only a standalone training notebook.

## 15. Project Workflow Summary

The full project workflow implemented in this repository is:

1. Generate or load wastewater dataset.
2. Clean and prepare the data.
3. Split data into training and testing sets.
4. Scale all numeric features.
5. Create supervised time-window sequences.
6. Train the CNN-LSTM-Attention model.
7. Evaluate the model using MAE, RMSE, R2, and MAPE.
8. Save trained models, plots, and reports.
9. Predict future water-quality values.
10. Serve predictions through FastAPI.
11. Consume predictions through the Flutter client.

## 16. Final Outcome

The final outcome of this project is a complete wastewater quality prediction pipeline that combines:

- data preprocessing
- CNN-based local feature extraction
- LSTM-based temporal learning
- attention-based importance weighting
- quantitative evaluation
- future forecasting
- API deployment
- mobile application integration

In summary, the project successfully demonstrates how a hybrid deep learning model can be used for STP parameter prediction and can be extended into a practical smart monitoring and forecasting solution.

## 17. Conclusion

This project satisfies all major items described in the PDF summary:

- model used
- languages and frameworks
- data-processing libraries
- visualization tools
- development environment context
- dataset description
- preprocessing methods
- training setup
- evaluation metrics
- final system outcome

In addition, the repository goes beyond the PDF summary by including a working backend API, future prediction module, saved models, generated reports, and a Flutter-based client application.