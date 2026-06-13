from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator

# GCP Configuration (These can also be pulled from Airflow Variables)
PROJECT_ID = "project-edd02cbd-005a-46d0-9ae"
REGION = "us-central1"
JOB_NAME = "bank-ml-job"

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
    'bank_marketing_cloud_run_pipeline',
    default_args=default_args,
    description='Serverless MLOps Pipeline using Cloud Run',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'gcp', 'cloud-run'],
)

with dag:
    
    # 1. Data Processing Task
    data_processing_task = CloudRunExecuteJobOperator(
        task_id='data_processing',
        project_id=PROJECT_ID,
        region=REGION,
        job_name=JOB_NAME,
        overrides={
            "container_overrides": [
                {
                    "args": ["python", "pipelines/data_pipeline.py"]
                }
            ]
        },
        deferrable=False,  # Turned off because Composer Triggerer is disabled
    )

    # 2. Model Training Task
    model_training_task = CloudRunExecuteJobOperator(
        task_id='model_training',
        project_id=PROJECT_ID,
        region=REGION,
        job_name=JOB_NAME,
        overrides={
            "container_overrides": [
                {
                    "args": ["python", "pipelines/training_pipeline.py"]
                }
            ]
        },
        deferrable=False,
    )

    # 3. Batch Inference Task
    batch_inference_task = CloudRunExecuteJobOperator(
        task_id='batch_inference',
        project_id=PROJECT_ID,
        region=REGION,
        job_name=JOB_NAME,
        overrides={
            "container_overrides": [
                {
                    "args": ["python", "pipelines/inference_pipeline.py"]
                }
            ]
        },
        deferrable=False,
    )

    # Define execution order
    data_processing_task >> model_training_task >> batch_inference_task
