# 1. Base Image: Use official minimal Python 3.11 Linux runtime
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install operating system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code, tracking database, and models
COPY src/ ./src/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

# 6. Expose the port Uvicorn listens on
EXPOSE 8000

# 7. Start the API server
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]