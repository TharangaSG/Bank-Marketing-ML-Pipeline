from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Add project root to path
sys.path.append('/home/tharangasg/projects/ML/Bank ML')

from pipelines.data_pipeline import data_pipeline
from pipelines.training_pipeline import training_pipeline
from pipelines.inference_pipeline import inference_pipeline

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2026, 1, 1),
}

dag = DAG(
    'bank_marketing_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML Pipeline for Bank Marketing',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'bank-marketing'],
)

# Wrapper functions for the Python operators
def run_data_processing(**kwargs):
    # Determine if we should use PySpark for data processing based on config or kwargs
    # For now, default to pandas, but allow overriding
    data_pipeline()

def run_model_training(**kwargs):
    training_pipeline()

def run_batch_inference(**kwargs):
    inference_pipeline()

with dag:
    
    # Optional: setup task to sync dependencies if needed
    setup_env = BashOperator(
        task_id='setup_environment',
        bash_command='cd "/home/tharangasg/projects/ML/Bank ML" && uv sync'
    )

    data_processing_task = PythonOperator(
        task_id='data_processing',
        python_callable=run_data_processing,
    )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
    )

    batch_inference_task = PythonOperator(
        task_id='batch_inference',
        python_callable=run_batch_inference,
    )

    # Define dependencies
    setup_env >> data_processing_task >> model_training_task >> batch_inference_task
