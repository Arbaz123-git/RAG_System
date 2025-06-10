FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt /app/
COPY api/requirements.txt /app/api_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy application code
COPY . /app/

# Make entrypoint script executable
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Set the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"] 