# Script to train machine learning model.

from sklearn.model_selection import train_test_split

import pandas as pd
import os
import numpy as np
import joblib

from ml.data import process_data
from ml.model import train_model, save_model, load_model, inference, compute_model_metrics, calculate_slice_metrics

# Add code to load in the data.

data = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "../data/census.csv"))

# Select interesting features
feature_list = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "salary"
]

data = data[feature_list]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2)

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

# Specify save folder
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one directory to the project directory
project_dir = os.path.abspath(os.path.join(script_dir, '../'))

# Specify target folder
target_folder = os.path.join(project_dir, 'model')

# Create the directory if it not exists
os.makedirs(target_folder, exist_ok=True)

# Save encoders for later use in API requests
joblib.dump(encoder, os.path.join(target_folder, 'encoder.pkl'))
joblib.dump(lb, os.path.join(target_folder, 'lb.pkl'))

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# TESTS
print('Shape Tests:')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Train a model
model = train_model(X_train, y_train)

# Save model in target folder
save_model(model, target_folder)

# Specify load folder and load model
load_folder = target_folder
model = load_model(load_folder)

# Run inference
pred = inference(model, X_test)

# Calculate and print metrics
precision, recall, fbeta = compute_model_metrics(y_test, pred)
print('Precision: ' + str(np.round(precision, 2)) + ' Recall: ' +
      str(np.round(recall, 2)) + ' fbeta: ' + str(np.round(fbeta, 2)))

# Prepare categorical columns for slice metric results
test['label'] = lb.transform(test['salary'])
test['pred'] = pred

# Calculate and print data slice metrics
slice_results = calculate_slice_metrics(
    test, test['label'], test['pred'], ['education'])
with open('slice_output.txt', 'w') as file:
    file.write(str(slice_results))
