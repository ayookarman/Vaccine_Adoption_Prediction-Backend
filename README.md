# Vaccine Adoption Prediction Backend

This is the backend for the Vaccine Adoption Prediction system, built using Flask.

## Features
- Predicts the probability of vaccine adoption based on user inputs.
- Uses a trained machine learning model for inference.
- Provides a REST API endpoint for predictions.

## Tech Stack
- Python 3
- Flask
- scikit-learn
- Pandas
- NumPy

## Setup & Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Git

### Clone the Repository
```
git clone <your-repo-url>
cd <repo-name>
```
### Create a Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Running the Backend
```
python app.py
```
The backend will start at http://127.0.0.1:5000/

### API Endpoints
## Predict Vaccine Adoption
- Endpoint: /predict
- Method: POST
- Request Body: JSON with user input fields
- Response: JSON with vaccine adoption probabilities
 Example Request
  ```
  {
  "age_group": "18-34 Years",
  "education": "College Graduate",
  "race": "White",
  "sex": "Male",
  "income_poverty": "<= $75,000, Above Poverty",
  "marital_status": "Single",
  "rent_or_own": "Rent",
  "employment_status": "Employed",
  "health_insurance": 1,
  "household_children": 0
  }
  ```
   Example Response
  ```
  {
  "xyz_vaccine_probability": 0.27,
  "seasonal_vaccine_probability": 0.41
  }
  ```
### Deployment
## Deploy on Render (Recommended)
- Create a new repository on GitHub and push the backend code.
- Go to Render, create a new web service.
- Connect your GitHub repo and set the runtime environment to Python.
- Add requirements.txt for dependencies.
- Set the start command to python app.py.
- Deploy and get the API URL.
## Deploy on Railway
- Sign up on Railway.
- Create a new project and connect GitHub repo.
- Set environment variables if needed.
- Deploy and get the API URL.

### License
This project is licensed under the MIT License.


Let me know if you need any changes! 🚀
