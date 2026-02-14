# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data,
    data_preprocessing,
    build_save_model,
    load_model_elbow,
    build_dbscan_model,     # NEW: DBSCAN task
    evaluate_models          # NEW: Evaluation task
)
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for the DAG
default_args = {
    'owner': 'arjun_avadhani',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG instance
dag = DAG(
    'Airflow_Lab1_Modified',
    default_args=default_args,
    description='Modified Lab 1: Iris dataset with KMeans + DBSCAN comparison and Silhouette evaluation',
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'clustering', 'iris', 'kmeans', 'dbscan'],
)

# ============================================================
# Task 1: Load Data (Iris dataset from sklearn)
# ============================================================
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# ============================================================
# Task 2: Data Preprocessing (StandardScaler normalization)
# ============================================================
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# ============================================================
# Task 3a: Build & Save KMeans Model (original flow)
# ============================================================
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
)

# ============================================================
# Task 3b: Build DBSCAN Model (NEW - parallel to KMeans)
# ============================================================
build_dbscan_task = PythonOperator(
    task_id='build_dbscan_task',
    python_callable=build_dbscan_model,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# ============================================================
# Task 4: Load Model & Elbow Method (original flow)
# ============================================================
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)

# ============================================================
# Task 5: Evaluate & Compare Models (NEW)
# ============================================================
evaluate_models_task = PythonOperator(
    task_id='evaluate_models_task',
    python_callable=evaluate_models,
    op_args=[data_preprocessing_task.output, build_dbscan_task.output],
    dag=dag,
)

# ============================================================
# Task Dependencies
# ============================================================
# Original flow:  load -> preprocess -> build_kmeans -> elbow
# New branch:     preprocess -> build_dbscan -> evaluate
# Both converge:  elbow + evaluate run after their respective builds
#
#                        +--> build_kmeans --> load_model_elbow
# load_data -> preprocess |
#                        +--> build_dbscan --> evaluate_models
#

load_data_task >> data_preprocessing_task
data_preprocessing_task >> build_save_model_task >> load_model_task
data_preprocessing_task >> build_dbscan_task >> evaluate_models_task

# Allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()