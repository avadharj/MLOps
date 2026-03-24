# Breast Cancer Prediction App

A machine learning web application that predicts whether a breast tumor is **malignant** or **benign** based on 30 numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Uses the [Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from scikit-learn with a Random Forest classifier.

## Watch the tutorial video at [Tutorial Video](https://www.youtube.com/watch?v=O0X6NoQyEf0)

## Project Structure

- `train.py` — Loads the breast cancer dataset, trains a Random Forest model, and saves it to `model/model.pkl`
- `predict.py` — Loads the saved model and exposes a `predict_cancer()` function
- `main.py` — Flask API that accepts 30 features via POST and returns a diagnosis prediction
- `test_api.py` — Sample script to test the `/predict` endpoint locally
- `streamlit_app.py` — Frontend UI for interacting with the deployed model
- `Dockerfile` — Container definition for Cloud Run deployment

## Input Features

The model expects 30 numeric features grouped into three categories (mean, standard error, and worst/largest):

**Mean values:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension

**Standard error:** same 10 measurements

**Worst (largest):** same 10 measurements

See `main.py` for the full list of feature names expected by the API.

## Local Setup

1. Train the model:
```
python train.py
```

2. Start the Flask server:
```
python main.py
```

3. Test the API:
```
python test_api.py
```

## Deploy to Google Cloud

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) based on your operating system and make sure the `gcloud` command works.

```
gcloud init
```

Make sure you have authenticated with the correct email and selected the correct project and region:

```
gcloud auth login
```

Enable the required APIs:

1. **Artifact Registry** — Create a repo named `gcr.io`
2. **Cloud Build**

```
gcloud services enable cloudbuild.googleapis.com
```

Build the Docker image using Cloud Build:

```
gcloud builds submit --tag gcr.io/[YOUR_PROJ_ID]/cancer-app
```

Deploy the container to Cloud Run (enable Cloud Run if this is your first time):

```
gcloud run deploy cancer-app --image gcr.io/[YOUR_PROJ_ID]/cancer-app --platform managed --port 8080 --allow-unauthenticated
```

Once the application is deployed, update the deployed URL in your frontend source code (`streamlit_app.py`), then run:

```
streamlit run streamlit_app.py
```
