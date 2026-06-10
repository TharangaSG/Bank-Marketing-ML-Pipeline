# Use official Python lightweight image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies (required for some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazing fast dependency installation
RUN pip install --no-cache-dir uv

# Copy only dependency definitions first to leverage Docker cache
COPY pyproject.toml .

# Install dependencies using uv directly into the system python
RUN uv pip install --system .

# Copy the rest of the application code
COPY . .

# Default command (Airflow will override this dynamically)
CMD ["python", "pipelines/data_pipeline.py"]
