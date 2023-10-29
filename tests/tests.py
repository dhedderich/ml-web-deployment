import numpy as np
import pytest
import joblib
import os
from pydantic import BaseModel
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, load_model, save_model
from main import app


def test_train_model_return_type():
    # Generate some sample training data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    # Call the train_model function
    model = train_model(X_train, y_train)

    # Check if the returned model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


def test_save_model_return_type(tmpdir):
    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one directory to the project directory
    project_dir = os.path.abspath(os.path.join(script_dir, '../'))

    # Specify load folder
    load_folder = os.path.join(project_dir, 'model')

    # load the model
    model = joblib.load(os.path.join(load_folder, 'model.pkl'))

    # Specify a temporary directory for testing
    test_folder = tmpdir.mkdir("test_data")
    test_folder = test_folder.mkdir('model')
    model_path = str(test_folder)

    # Call the save_model function
    # model_path = str(test_folder.join("model.pkl"))
    save_model(model, model_path)

    # Check if the model file was saved
    assert os.path.isfile(f'{model_path}/model.pkl')

    # Load the saved model
    loaded_model = joblib.load(f'{model_path}/model.pkl')

    # Check if the loaded model is equal to the original model
    assert isinstance(loaded_model, RandomForestClassifier) == isinstance(
        model, RandomForestClassifier)


def test_load_model_return_type(tmpdir):
    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one directory to the project directory
    project_dir = os.path.abspath(os.path.join(script_dir, '../'))

    # Specify load folder
    load_folder = os.path.join(project_dir, 'model')

    # load the model
    model = joblib.load(os.path.join(load_folder, 'model.pkl'))

    # Specify a temporary directory for testing
    test_folder = tmpdir.mkdir('test_data')
    test_folder = test_folder.mkdir('model')
    model_path = str(test_folder)

    # Save the model
    joblib.dump(model, os.path.join(model_path, 'model.pkl'))

    # Call the load_model function to load the model
    loaded_model = load_model(model_path)

    # Check if the loaded model is an instance of the expected type
    assert isinstance(loaded_model, type(model))

# FastAPI Tests


client = TestClient(app)


def test_GET_endpoint():
    # Send a GET request to the "/welcome" endpoint
    response = client.get("/")

    # Check that the response status code is 200
    assert response.status_code == 200

    # Assert that the response contains the expected message
    expected_message = {
        "Welcome": "Welcome to one of the finest classifiers available"}
    assert response.json() == expected_message

# Declare the post object


class RequestItem(BaseModel):
    # Define the structure of the input data for testing
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str

# Declare the response object


class PredictionResponse(BaseModel):
    prediction: float


def test_POST_inference_endpoint_successful():
    # Define test input data
    input_data = {
        "inference": {
            "workclass": "Private",
            "education": "HS-grad",
            "marital_status": "Divorced",
            "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "native_country": "United-States",
        },
        "response_model": {
            "prediction": 0
        }
    }

    # Send a POST request to the "/inference" endpoint with the test input data
    response = client.post("/inference/", json=input_data)
    print(response.text)
    # Assert that the response status code is 200
    assert response.status_code == 200

    # Assert that the response contains a "prediction" key
    response_data = response.json()
    assert "prediction" in response_data
    assert 0 == response_data["prediction"]


def test_POST_inference_endpoint_error_handling():
    # Define test input data with incorrect values
    input_data = {
        "inference": {
            "workclass": "Private",
            "education": "HS-grad",
            "marital_status": "Divorced",
            "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Other-Gender",
            "native_country": "United-States",
        },
        "response_model": {
            "prediction": 0
        }
    }

    # Send a POST request to the "/inference" endpoint with test input data
    response = client.post("/inference/", json=input_data)

    # Assert that the response status code is 500 
    assert response.status_code == 500
