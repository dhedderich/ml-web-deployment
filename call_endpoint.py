import requests

# Define correct URL
api_url = "https://ml-web-deployment.onrender.com/inference/"

# Define the data to send in the POST request
data = {
  "workclass": "Private",
  "education": "HS-grad",
  "marital_status": "Divorced",
  "occupation": "Handlers-cleaners",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "native_country": "United-States"
}

# Send a POST request to the API
response = requests.post(api_url, json=data)

# Check the status code
status_code = response.status_code

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
else:
    result = None

# Print the status code and result
print("Status Code:", status_code)
print("Result:", result)