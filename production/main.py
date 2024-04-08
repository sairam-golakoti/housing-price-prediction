import os
import sys
import mlflow
import subprocess
import os.path as op

from ta_lib.core.api import (
    create_context,
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_DATA_BASE_PATH,
)

sys.path.append("./production")

from scoring import score_model
from training import train_model
from feature_engineering import transform_features
from data_cleaning import clean_order_table,clean_product_table,clean_sales_table,create_training_datasets

HERE = op.dirname(op.abspath(__file__))

if __name__ == "__main__":

    expt_name = "TAMLEP_MLFlow"
    tracking_uri = "http://127.0.0.1:8082"

    config_path = os.path.join("./production", "conf", "config.yml")
    context = create_context(config_path)

    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    try:
        expt = client.get_experiment_by_name(expt_name)
        expt_id = expt.experiment_id

    except Exception as e:
        print("Error getting Ml Flow experiment: ", e)
        expt_id = client.create_experiment(expt_name)


    with mlflow.start_run(experiment_id=expt_id, run_name="Sales Price Prediction") as parent_run:

        with mlflow.start_run(experiment_id=expt_id, run_name="Data Cleaning", nested=True) as run:
            # out = subprocess.run([sys.executable, "./production/data_cleaning.py"], capture_output=True, text=True)

            clean_prod = clean_product_table(context, {})
            clean_ordr = clean_order_table(context, {})
            clean_sals = clean_sales_table(context, {})
            create_training_datasets(context, {"test_size":0.2, "target":"unit_price"})

            output_train_features = DEFAULT_DATA_BASE_PATH + "/train/sales/features.parquet"
            output_train_target = DEFAULT_DATA_BASE_PATH + "/train/sales/target.parquet"
            output_test_features = DEFAULT_DATA_BASE_PATH + "/test/sales/features.parquet"
            output_test_target = DEFAULT_DATA_BASE_PATH + "/test/sales/target.parquet"

            mlflow.log_params({
                "Product Table rows": clean_prod.shape[0],
                "Product Table attributes": clean_prod.shape[1],
                "Order Table rows": clean_ordr.shape[0],
                "Order Table attributes": clean_ordr.shape[1],
                "Sales Table rows": clean_sals.shape[0],
                "Sales Table attributes": clean_sals.shape[1],
                "Training data exist": os.path.isfile(output_train_features) and os.path.isfile(output_train_target),
                "Test data exist": os.path.isfile(output_test_features) and os.path.isfile(output_test_target),
            })

        with mlflow.start_run(experiment_id=expt_id, run_name="Feature Engineering", nested=True):
            # out = subprocess.run([sys.executable, "./production/feature_engineering.py"], capture_output=True, text=True)
            transform_features(context, params={"outliers": {"method": "mean","drop": False }})
            mlflow.log_params({
                "Curated columns Pipeline exist": os.path.isfile(op.abspath(op.join(DEFAULT_ARTIFACTS_PATH, "curated_columns.joblib"))),
                "Feature Engineering Pipeline exist": os.path.isfile(op.abspath(op.join(DEFAULT_ARTIFACTS_PATH, "features.joblib")))
            })

        with mlflow.start_run(experiment_id=expt_id, run_name="Model Training", nested=True):
            # out = subprocess.run([sys.executable, "./production/training.py"], capture_output=True, text=True)
            train_model(context,{})
            mlflow.log_param("Model Training Pipeline exist", os.path.isfile(op.abspath(op.join(DEFAULT_ARTIFACTS_PATH, "train_pipeline.joblib"))))

        with mlflow.start_run(experiment_id=expt_id, run_name="Model Scoring", nested=True):
            # out = subprocess.run([sys.executable, "./production/scoring.py"], capture_output=True, text=True)
            res = score_model(context, {})
            mlflow.log_param("RMSE Value", res)
