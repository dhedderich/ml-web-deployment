import numpy as np
import pytest
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, load_model, save_model

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
    #model_path = str(test_folder.join("model.pkl"))
    save_model(model, model_path)

    # Check if the model file was saved
    assert os.path.isfile(f'{model_path}/model.pkl')

    # Load the saved model
    loaded_model = joblib.load(f'{model_path}/model.pkl')

    # Check if the loaded model is equal to the original model
    assert isinstance(loaded_model, RandomForestClassifier) == isinstance(model, RandomForestClassifier)
    
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