from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starter.ml.model import load_model
from starter.ml.data import process_data
import pandas as pd
import joblib
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(filename="fastapi.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Declare the data request object


class RequestItem(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "workclass": "Private",
                    "education": "HS-grad",
                    "marital_status": "Divorced",
                    "occupation": "Handlers-cleaners",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "native_country": "United-States",
                }
            ]
        }
    }

# Declare the response object


class PredictionResponse(BaseModel):
    prediction: float


# Create app
app = FastAPI()

# GET request for greeting


@app.get("/")
async def welcome():
    return {"Welcome": "Welcome to one of the finest classifiers available"}

# POST request for inference


@app.post("/inference/")
async def inference(inference: RequestItem,
                    response_model: PredictionResponse):
    try:
        df_request = pd.DataFrame([inference.dict()])

        # Locate folder of encoder & model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_folder = os.path.join(script_dir, 'model')
        encoder = joblib.load(os.path.join(target_folder, 'encoder.pkl'))
        lb = joblib.load(os.path.join(target_folder, 'lb.pkl'))

        # Fix naming "_" for "-" as the encoder was trained this way
        df_request.columns = df_request.columns.str.replace('_', '-')

        # Prepare data for respective encoder
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        # Reorder the columns based on the specific order
        df_ordered = df_request[cat_features]

        # Preprocess the data
        df_prepared, y, encoder, lb = process_data(
            df_ordered, categorical_features=cat_features,
            label=None, training=False, encoder=encoder, lb=lb)
        logger.info("DataFrame shape: %s", df_prepared.shape)

        # Load model and perform inference
        model = load_model(target_folder)
        result = model.predict(df_prepared)
        logger.info("Result: %s", result[0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    if isinstance(result, np.ndarray):
        result = result[0].astype(float)
    return {"prediction": result}
