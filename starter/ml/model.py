from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib, os

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a random forest machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
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

def load_model(path: str = 'model'):
    """Loads a machine learning model at a specified location

    Args:
        path (str): optional path to load the model
    Returns:
        model: Trained machine learning model
    """
    # load the model
    model = joblib.load(os.path.join(path, 'model.pkl'))

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Calculate metrics
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Run inference
    pred = model.predict(X)
    return pred
    
