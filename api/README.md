# MultiModal RAG API

A secure REST API for the MultiModal RAG system with JWT authentication and two-tier LRU caching.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv api-env
source api-env/bin/activate  # On Windows: api-env\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the following content:

```
# API Keys
GROQ_API_KEY=your_groq_api_key_here

# JWT Settings
JWT_SECRET_KEY=your_jwt_secret_key_for_production_use_a_strong_random_string

# Cache Settings (optional)
MEMORY_CACHE_SIZE=5000
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=86400
```

3. Start Redis (optional, but recommended for two-tier caching):

```bash
docker-compose up -d redis
```

4. Start the API server:

```bash
cd api
python run_api_with_cache.py
```

The API will be available at http://localhost:8000.

## Caching Layer

The API includes a two-tier LRU caching system:

1. **In-Memory LRU Cache**: A fast, fixed-size dictionary in memory that evicts least recently used items when full.
2. **Redis Cache**: A persistent cache with TTL-based expiration for longer-term storage.

### Cache Configuration

The caching system can be configured using environment variables:

- `MEMORY_CACHE_SIZE`: Maximum number of entries in the in-memory cache (default: 5000)
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis password (default: None)
- `REDIS_TTL`: Time-to-live for Redis cache entries in seconds (default: 86400, 24 hours)

### Cache Behavior

- When a query is received, the system first checks the in-memory cache
- If not found, it checks the Redis cache
- If still not found, it processes the query and stores the result in both caches
- The API response includes a `cache_hit` field in the metadata that indicates the source of the response ("memory", "redis", or "miss")

### Testing and Benchmarking

The repository includes scripts for testing and benchmarking the cache:

```bash
# Run unit tests for the cache implementation
python -m unittest api.test_cache

# Run a benchmark to measure cache performance
python api/benchmark_cache.py
```

## API Endpoints

### GET /

Returns basic information about the API.

### POST /token

Exchange credentials for a JWT token.

**Request:**

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=clinician1&password=secret1"
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### POST /ask

Ask a question to the MultiModal RAG system.

**Request:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -d '{"query": "What are the characteristics of polyps in colonoscopy images?"}'
```

**Response:**

```json
{
  "answer": "Based on the retrieved information, polyps in colonoscopy images appear as...",
  "metadata": {
    "timestamp": "2023-06-15T12:34:56.789012",
    "user": "clinician1",
    "sources": ["Polyps in colonoscopy images often appear as raised, rounded structures protruding from the intestinal wall..."],
    "image_sources": ["images/colonoscopy_polyp_01.jpg"],
    "cache_hit": "miss"
  }
}
```

## Security Notes

- JWT tokens expire after 60 minutes
- The API uses HS256 for JWT signing
- In production, use a strong random string for JWT_SECRET_KEY
- In production, implement proper password hashing (e.g., bcrypt)
- In production, use a real database for user management
- In production, specify allowed origins for CORS 