#!/bin/bash
set -e

# Wait for Weaviate to be ready
echo "Waiting for Weaviate to be ready..."
until curl -s -f "http://${WEAVIATE_URL:-weaviate:8080}/v1/.well-known/ready" > /dev/null; do
  echo "Weaviate is not ready yet. Retrying in 5 seconds..."
  sleep 5
done
echo "Weaviate is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
until redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping > /dev/null; do
  echo "Redis is not ready yet. Retrying in 5 seconds..."
  sleep 5
done
echo "Redis is ready!"

# Check if Weaviate schema exists
echo "Checking if Weaviate schema exists..."
SCHEMA_EXISTS=$(curl -s "http://${WEAVIATE_URL:-weaviate:8080}/v1/schema" | grep -c "TextEmbeddings" || true)
if [ "$SCHEMA_EXISTS" -eq "0" ]; then
  echo "Creating Weaviate schema..."
  python create_weaviate_schema.py
  echo "Schema created successfully!"
else
  echo "Weaviate schema already exists."
fi

# Check if we need to upload embeddings
echo "Checking if embeddings need to be uploaded..."
CLASS_COUNT=$(curl -s "http://${WEAVIATE_URL:-weaviate:8080}/v1/objects?class=TextEmbeddings&limit=1" | grep -c "id" || true)
if [ "$CLASS_COUNT" -eq "0" ]; then
  echo "Uploading embeddings to Weaviate..."
  python upload_embeddings.py
  echo "Embeddings uploaded successfully!"
else
  echo "Embeddings already exist in Weaviate."
fi

# Start the API
echo "Starting the MultiModal RAG API..."
exec python api/run_api_with_cache.py 