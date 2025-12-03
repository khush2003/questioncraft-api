# Use the official Python image as a base
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --requirement requirements.txt --upgrade --upgrade-strategy only-if-needed --no-warn-script-location --no-warn-conflicts --force-reinstall --no-compile --disable-pip-version-check --no-cache-dir --user

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]