# Census Income Prediction with Random Forest

## Overview
This repository contains code to build a machine learning model using the Random Forest algorithm. The model is trained on the "Census Income" dataset, also known as the "Adult" dataset. The goal is to predict whether an individual's income exceeds $50,000 per year based on various census attributes.

You can find a detailed description of the dataset [here](https://archive.ics.uci.edu/ml/datasets/census+income).

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Training the Model](#training-the-model)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [FastAPI - RESTful API](#fastapi---restful-api)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
The repository is structured as follows:
- `train_model.py`: Python script to create and train the Random Forest model.
- `requirements.txt`: List of required Python libraries.
- `.github/workflows/`: GitHub Actions workflows for linting (flake8), unit testing (pytest), and installing requirements.
- `main.py`: FastAPI definition for POST (inference) and GET method

## Getting Started
### Prerequisites
Before you start, make sure you have the following tools and packages installed:
- Python
- Pip
- Virtual environment (recommended)

To install the required Python libraries, run:

```
pip install -r requirements.txt

```

### Training the Model
To create and train the Random Forest model, use the following command in the root of the repository:

```
python train_model.py

```
To analyze the ML model's prediction errors there is a function called "calculate_slice_metrics" in `model.py` that computes the data slice metrics (precision, recall, fbeta) of categorical columns' unique classes of the test set.

## Testing
Unit tests for the code are implemented using Pytest. You can run the tests within the /tests directory using the following command:

```
pytest tests.py

```

## Continuous Integration
This repository is set up with GitHub Actions for a basic Continuous Integration/Continuous Deployment (CI/CD) pipeline. The following checks are performed on every push:
- Flake8 linter checks for code style.
- pytest runs unit tests.
- Requirements are installed to ensure dependencies are up to date.

## FastAPI - RESTful API
In addition to the machine learning model, a FastAPI-based RESTful API is included. You can find the API in the `main.py` file in the root directory. It has the following endpoints:

### Greeting Endpoint (GET)
A simple GET request that greets the user.

### Prediction Endpoint (POST)
This endpoint loads the trained model and provides predictions. To receive a Code 200 response, use a JSON payload with the following format:
```json
{
  "workclass": "Private",
  "education": "HS-grad",
  "marital_status": "Divorced",
  "occupation": "Handlers-cleaners",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "native_country": "United-States"
}
```

The prediction is given back to the requestor in the following format:
```json
{
   "prediction": 0.0
}
```
You can run the API locally via:

```
uvicorn main:app
```

You can call a currently running endpoint on Render with the code in the `call_endpoint.py` file.

## Contributing
Feel free to contribute to this project. You can fork the repository, make your changes, and create a pull request. Please ensure your code follows the established style guide.

## License
This project is licensed under the MIT License. You are free to use and modify the code for your needs.
