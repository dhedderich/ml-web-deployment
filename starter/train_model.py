# Script to train machine learning model.

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import joblib
from ml.data import process_data

# Add code to load in the data.

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/census.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)

# TESTS
print(X_train.shape)
print(y_train.shape)

# Train a model and save

rf = RandomForestClassifier()
model = rf.fit(X_train, y_train)

folder_path = os.path.join(os.path.dirname(__file__), "../model")

joblib.dump(model, f'{folder_path}/rf_model.pkl')

