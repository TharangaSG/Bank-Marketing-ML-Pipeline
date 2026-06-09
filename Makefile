.PHONY: install data train inference all mlflow clean

# Python environment settings using uv
PYTHON_ENV = uv run python
MLFLOW_URI = sqlite:///mlflow.db

# Help menu
help:
	@echo "Available commands:"
	@echo "  make install    - Install project dependencies using uv"
	@echo "  make data       - Run the data processing pipeline"
	@echo "  make train      - Run the model training pipeline"
	@echo "  make inference  - Run the batch inference pipeline"
	@echo "  make all        - Run both data processing and training pipelines"
	@echo "  make mlflow     - Start the MLflow UI server"
	@echo "  make clean      - Remove generated artifacts, pycache, and MLflow DB"

# Install dependencies
install:
	uv sync

# Run data processing pipeline
data:
	$(PYTHON_ENV) pipelines/data_pipeline.py

# Run model training pipeline
train:
	$(PYTHON_ENV) pipelines/training_pipeline.py

# Run batch inference pipeline
inference:
	$(PYTHON_ENV) pipelines/inference_pipeline.py

# Run end-to-end (data + train)
all: data train

# Start MLflow tracking UI
mlflow:
	uv run mlflow ui --backend-store-uri $(MLFLOW_URI) --port 5000

# Clean up generated artifacts and cached files
clean:
	@echo "Cleaning up project..."
	rm -rf artifacts/data/*
	rm -rf artifacts/models/*
	rm -rf artifacts/predictions/*
	rm -rf artifacts/train/*
	rm -rf data/processed/*
	rm -rf mlruns/
	rm -f mlflow.db
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cleanup complete!"
