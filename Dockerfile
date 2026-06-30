# Use an official Python runtime as a parent image (CPU only)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the model_output directory just in case it doesn't exist
RUN mkdir -p model_output

# Define environment variables to enforce deterministic behavior and prevent bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Specify the command to run on container start
# This will execute the ranking pipeline and produce model_output/submission.csv
CMD ["python", "main.py"]
