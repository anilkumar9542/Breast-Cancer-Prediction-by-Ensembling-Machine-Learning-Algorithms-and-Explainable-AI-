from flask import Flask, render_template, request
import numpy as np
import pickle
import shap
import pandas as pd

app = Flask(__name__, static_url_path='/static')

# Load the trained model
ensemble_model = pickle.load(open('ensemble_classifier(75-25).pkl', 'rb'))

# Extract the individual tree-based models from the ensemble
random_forest_model = ensemble_model.estimators_[0]

gradient_boosting_model = ensemble_model.estimators_[1]

# Create individual SHAP explainer objects for each tree-based model
explainer_rf = shap.TreeExplainer(random_forest_model)

explainer_gb = shap.TreeExplainer(gradient_boosting_model)

shap.initjs()

@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input features
        features = ["texture_mean", "smoothness_mean", "compactness_mean", "symmetry_mean", 
                    "fractal_dimension_mean", "texture_se", "smoothness_se", "symmetry_se", "symmetry_worst"]
        input_values = [float(request.form[feature]) for feature in features]
        
        values = np.array([input_values])
        
        # Make prediction
        prediction = ensemble_model.predict(values)
        # Explain the prediction using SHAP values
        shap_values = explainer_rf.shap_values(values)

        # Convert shap_values to a dictionary with feature names
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                shap_values = shap_values[0]
            else:
                shap_values = np.array(shap_values)  # Convert to array if it's still a list

        # Ensure shap_values has the same shape as values
        if shap_values.shape[0] != values.shape[1]:
            shap_values = shap_values.T

        feature_shap_values = {feature: shap_values[i] for i, feature in enumerate(features)}

        return render_template('result.html', prediction=prediction[0], feature_shap_values=feature_shap_values)

if __name__ == "__main__":
    app.run(debug=True)
