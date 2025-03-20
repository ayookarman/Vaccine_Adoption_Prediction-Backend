import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Define relative paths for dataset
DATA_FOLDER = "./data"
train_features_path = os.path.join(DATA_FOLDER, "training_set_features.csv")
train_labels_path = os.path.join(DATA_FOLDER, "training_set_labels.csv")
test_features_path = os.path.join(DATA_FOLDER, "test_set_features.csv")
submission_path = os.path.join(DATA_FOLDER, "submission_format.csv")

# Load datasets
train_features = pd.read_csv(train_features_path)
train_labels = pd.read_csv(train_labels_path)
test_features = pd.read_csv(test_features_path)
submission_format = pd.read_csv(submission_path)

# Ensure correct column names in training labels
expected_columns = {'xyz_vaccine', 'seasonal_vaccine'}
actual_columns = set(train_labels.columns)
missing_columns = expected_columns - actual_columns

if missing_columns:
    raise KeyError(f"❌ Error: Missing columns in training_set_labels.csv -> {missing_columns}")

# Separate categorical and numerical features
categorical_features = train_features.select_dtypes(include=['object']).columns
numerical_features = train_features.select_dtypes(include=['number']).columns.drop('respondent_id')

# Handle missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Impute numerical features
train_features[numerical_features] = num_imputer.fit_transform(train_features[numerical_features])
test_features[numerical_features] = num_imputer.transform(test_features[numerical_features])

# Impute and encode categorical features
imputed_cat_train = cat_imputer.fit_transform(train_features[categorical_features])
encoded_cat_train = encoder.fit_transform(imputed_cat_train)
imputed_cat_test = cat_imputer.transform(test_features[categorical_features])
encoded_cat_test = encoder.transform(imputed_cat_test)

# Combine processed features
processed_train_features = np.hstack((train_features[numerical_features], encoded_cat_train))
processed_test_features = np.hstack((test_features[numerical_features], encoded_cat_test))

# Prepare labels
labels = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

# Split data
X_train, X_val, y_train, y_val = train_test_split(processed_train_features, labels, test_size=0.2, random_state=42)

# Train model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_rf = MultiOutputClassifier(rf_classifier, n_jobs=-1)
multi_target_rf.fit(X_train, y_train)

# Predict on test set
test_predictions = multi_target_rf.predict_proba(processed_test_features)
test_pred_proba = np.array([pred[:, 1] for pred in test_predictions]).T

# Prepare submission
submission_format['xyz_vaccine'] = test_pred_proba[:, 0]
submission_format['seasonal_vaccine'] = test_pred_proba[:, 1]
submission_format.to_csv(os.path.join(DATA_FOLDER, "submission.csv"), index=False)

print("✅ Submission file created successfully!")

# ---------- Flask Routes ----------

@app.route("/")
def home():
    return jsonify({"message": "Vaccine Adoption Prediction API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Takes JSON input for a single user's data and returns vaccine adoption probabilities.
    """
    data = request.json
    try:
        input_df = pd.DataFrame([data])
        
        # Process input
        input_df[numerical_features] = num_imputer.transform(input_df[numerical_features])
        imputed_cat_input = cat_imputer.transform(input_df[categorical_features])
        encoded_cat_input = encoder.transform(imputed_cat_input)
        processed_input = np.hstack((input_df[numerical_features], encoded_cat_input))

        # Make prediction
        prediction = multi_target_rf.predict_proba(processed_input)
        prediction_proba = [pred[:, 1][0] for pred in prediction]

        return jsonify({
            "xyz_vaccine_probability": prediction_proba[0],
            "seasonal_vaccine_probability": prediction_proba[1]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Takes JSON input for multiple users and returns vaccine adoption probabilities.
    """
    data = request.json
    try:
        input_df = pd.DataFrame(data)
        
        # Process input
        input_df[numerical_features] = num_imputer.transform(input_df[numerical_features])
        imputed_cat_input = cat_imputer.transform(input_df[categorical_features])
        encoded_cat_input = encoder.transform(imputed_cat_input)
        processed_input = np.hstack((input_df[numerical_features], encoded_cat_input))

        # Make predictions
        predictions = multi_target_rf.predict_proba(processed_input)
        prediction_proba = np.array([pred[:, 1] for pred in predictions]).T

        response = [
            {"xyz_vaccine_probability": prob[0], "seasonal_vaccine_probability": prob[1]}
            for prob in prediction_proba
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
