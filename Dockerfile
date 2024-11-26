# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file (if you have one)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to container
COPY . .

# Expose port 5001
EXPOSE 5001

# Command to run the application
CMD ["python", "app.py", "--port=5001"]