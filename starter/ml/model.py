from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd


def train_model(X_train, y_train):
    """
    Trains a random forest machine learning model and returns it.

    Args:
        X_train : Training data
        y_train : Labels of training data
    Returns:
        model : Trained machine learning model.
    """
    # Create model
    rf = RandomForestClassifier()

    # Train model
    model = rf.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    """Saves a machine learning model at a specified location

    Args:
        model : machine learning model to save
        path (str): optional path to save the model
    """
    # Save the model
    joblib.dump(model, os.path.join(path, 'model.pkl'))


def load_model(path: str):
    """Loads a machine learning model at a specified location

    Args:
        path (str): optional path to load the model
    Returns:
        model : Trained machine learning model
    """
    # load the model
    model = joblib.load(os.path.join(path, 'model.pkl'))

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning
    model using precision, recall and fbeta.

    Args:
        y : Known labels
        preds : Predicted labels
    Returns:
        precision : precision metric result
        recall : recall metric result
        fbeta : fbeta metric result
    """
    # Calculate metrics
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Args:
        model : Trained machine learning model
        X : Input data used for prediction.
    Returns:
        preds : Inference from the model
    """
    # Run inference
    pred = model.predict(X)
    return pred


def calculate_slice_metrics(X_test, y_test, pred, column_names):
    """ Calculate precision, recall and fbeta for slices of all inputed columns

    Args:
        X_test : Input dataframe with all columns
        y_test : Label of the input dataframe
        pred : Prediction of the model for the respective data
        column_names : Columns to compute the metrics for
    Returns:
        result_dict : Dictionary with all metrics for
        each unique class per column
    """
    # Ensure X_test, y_test, and pred are pandas dataframes to handle them
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    pred = pd.DataFrame(pred)

    # Transform object types to category types for better manipulation
    object_columns = X_test.select_dtypes(include='object')
    X_test[object_columns.columns] = object_columns.astype('category')

    # Use column_names to calculate its metrics
    categorical_columns = X_test[column_names]

    # Initialize a dictionary to store results
    result_dict = {}

    # Iterate over categorical columns
    for column in categorical_columns:
        unique_classes = X_test[column].cat.categories
        for unique_class in unique_classes:
            # Filter rows based on the unique class
            y_true = y_test[y_test.index.isin(
                X_test[X_test[column] == unique_class].index)]
            y_pred = pred[pred.index.isin(y_true.index)]

            # Calculate metrics
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            fbeta = fbeta_score(y_true, y_pred, beta=1, average='weighted')

            # Store the results in the dictionary
            if column not in result_dict:
                result_dict[column] = {}
            result_dict[column][unique_class] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta
            }
    return result_dict
