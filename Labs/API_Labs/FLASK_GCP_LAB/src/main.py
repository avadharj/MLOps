from flask import Flask, request, jsonify
from predict import predict_cancer
import os

app = Flask(__name__)

# Map numeric model output to human-readable class
label_map = {
    0: "malignant",
    1: "benign"
}

# The 30 feature names expected by the model (from sklearn.datasets.load_breast_cancer)
FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract all 30 features from the request
    features = [float(data[name]) for name in FEATURE_NAMES]

    print("Input features:", features)

    # Call model
    prediction = predict_cancer(features)

    # Convert numeric class to label string for frontend
    try:
        pred_int = int(prediction)
        pred_label = label_map.get(pred_int, str(pred_int))
    except Exception:
        pred_label = str(prediction)

    return jsonify({'prediction': pred_label})

if __name__ == '__main__':
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
    