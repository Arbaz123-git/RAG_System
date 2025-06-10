# Docker Setup for MultiModal RAG System

This guide explains how to run the MultiModal RAG system with Docker for easy deployment and sharing.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- GROQ API key (for LLM integration)

## Quick Start

1. **Create an environment file**

   Create a file named `.env` in the project root with the following content:
   ```
   # API Keys
   GROQ_API_KEY=your_groq_api_key_here
   
   # JWT Settings
   JWT_SECRET_KEY=your_jwt_secret_key_for_production_use_a_strong_random_string
   
   # Cache Settings
   MEMORY_CACHE_SIZE=5000
   REDIS_TTL=86400
   ```
   
   Replace `your_groq_api_key_here` with your actual GROQ API key.

2. **Start the services**

   ```bash
   docker-compose up -d
   ```

   This will start:
   - Weaviate (vector database)
   - Redis (caching layer)
   - API service (MultiModal RAG API)

3. **Check that everything is running**

   ```bash
   docker-compose ps
   ```

## Testing the API

1. **Get an authentication token**

   ```bash
   curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=clinician1&password=secret1"
   ```

2. **Make a query**

   ```bash
   curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE" \
     -d '{"query": "What are the characteristics of polyps in colonoscopy images?"}'
   ```

## Load Testing

To run load tests with Locust, uncomment the Locust service in `docker-compose.yml` and run:

```bash
docker-compose up -d locust
```

Then open http://localhost:8089 in your browser to access the Locust web interface.

## Stopping the Services

```bash
docker-compose down
```

To remove all data volumes and start fresh:

```bash
docker-compose down -v
```

## Customization

### Scaling

To handle more traffic, you can adjust the cache settings in the `.env` file:

```
MEMORY_CACHE_SIZE=10000
REDIS_TTL=86400
```

### Persistent Data

Data is stored in Docker volumes:
- `weaviate_data`: Vector database storage
- `redis_data`: Redis cache data

Additionally, the following directories are mounted from your local system:
- `./data`: Dataset files
- `./embeddings`: Generated embeddings

## Troubleshooting

### Checking Logs

To see logs for a specific service:

```bash
docker-compose logs api
docker-compose logs weaviate
docker-compose logs redis
```

### Restarting a Service

If one service is having issues:

```bash
docker-compose restart api
```

### Container Shell Access

To get a shell inside a running container:

```bash
docker-compose exec api bash
``` 