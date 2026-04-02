# Streamlit Deployment

## Run Locally

From project root:

```bash
pip install -r requirements.txt
streamlit run deploy/streamlit/app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Select this repository and branch.
4. Set the app file path to:
   - `deploy/streamlit/app.py`
5. Deploy.

## Expected Inputs

Upload a CSV file that contains:
- Historical rows (at least `time_steps` rows, default 30)
- Numeric feature columns
- A target column matching one of:
  - `BOD (mg/L)`
  - `COD (mg/L)`
  - `PH`
  - `DO (mg/L)`

## Model Discovery

The app auto-selects model files from the project using these paths:
- `models/cnn_lstm_attention_model.h5`
- `models/bod_cnn_lstm_attention_model.h5`
- `models/cod_cnn_lstm_attention_model.h5`
- `models/ph_cnn_lstm_attention_model.h5`
- `models/do_cnn_lstm_attention_model.h5`
- `smart_stp_predictor/backend/models/bod_model.h5`
