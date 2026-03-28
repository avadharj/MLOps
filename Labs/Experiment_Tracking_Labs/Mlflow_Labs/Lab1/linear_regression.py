# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Optional CLI args:
    # 1) model_type: "rf" (default) or "elasticnet"
    # 2) dataset_type: "white" (default) or "red"
    # 3) alpha: ElasticNet alpha (default 0.5)
    # 4) l1_ratio: ElasticNet l1_ratio (default 0.5)
    model_type = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "rf"
    dataset_type = sys.argv[2].strip().lower() if len(sys.argv) > 2 else "white"
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    l1_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    dataset_urls = {
        "red": "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv",
        "white": "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    }

    if dataset_type not in dataset_urls:
        raise ValueError("dataset_type must be one of: red, white")

    # Read wine-quality csv file from the URL
    csv_url = dataset_urls[dataset_type]
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        raise

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    with mlflow.start_run():
        if model_type == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError("model_type must be one of: rf, elasticnet")

        model.fit(train_x, train_y.values.ravel())
        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        if model_type == "elasticnet":
            print(
                f"ElasticNet model on {dataset_type} wine (alpha={alpha:f}, l1_ratio={l1_ratio:f}):"
            )
        else:
            print(f"RandomForestRegressor model on {dataset_type} wine:")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("dataset_type", dataset_type)
        if model_type == "elasticnet":
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
        else:
            mlflow.log_param("n_estimators", 300)
            mlflow.log_param("max_depth", 12)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = model.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="WineQualityModelV2",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
