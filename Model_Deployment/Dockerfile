FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model server code
COPY app.py .

# Copy models directory
COPY ./Model_Training/models/ ./models/

# Copy necessary model code
COPY ./Model_Training/multimodel_train.py ./Model_Training/
COPY ./Model_Training/balanced_retraining.py ./Model_Training/
COPY ./Model_Training/__init__.py ./Model_Training/

# Create volume mount points
VOLUME ["/app/models"]

# Expose the port
EXPOSE 8000

# Set environment variables
ENV MODEL_DIR=/app/models
ENV DEFAULT_MODEL=linear_gbm_ensemble
ENV PORT=8000

# Run the FastAPI server
CMD ["python", "app.py"]
