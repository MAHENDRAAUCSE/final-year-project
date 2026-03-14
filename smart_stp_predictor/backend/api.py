from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from predictor import predict_future

app = FastAPI()

# Add CORS middleware to allow web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        result = predict_future(df)
        return {"future_predictions": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))