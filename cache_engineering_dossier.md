# Two-Tier LRU Caching Layer for MultiModal RAG

## Overview

This document describes the implementation of a two-tier LRU (Least Recently Used) caching system for the MultiModal RAG API. The caching system is designed to improve response times and reduce computational load by avoiding redundant processing of identical queries.

## Architecture

The caching system consists of two layers:

1. **In-Memory LRU Cache**: A fast, fixed-size dictionary in memory
2. **Redis Cache**: A secondary persistent layer with TTL-based expiration

### Cache Flow

When a request hits the `/ask` endpoint:

1. First, check the in-memory LRU cache
   - If hit: Return immediately
2. If miss, check the Redis cache
   - If hit: Populate the in-memory cache and return
3. If miss in both caches:
   - Execute the full RAG workflow
   - Store the result in both cache layers
   - Return the result

## Implementation Details

### In-Memory LRU Cache

The in-memory cache is implemented using Python's `collections.OrderedDict` which maintains insertion order. This allows for efficient implementation of LRU behavior:

- When an item is accessed, it's moved to the end of the OrderedDict (most recently used)
- When the cache reaches its maximum size, the first item (least recently used) is removed
- Default maximum size is 5,000 entries, configurable via environment variables

```python
class InMemoryLRUCache:
    def __init__(self, max_size=5000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        # Move to end (mark as most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def set(self, key, value):
        # If key exists, remove it first to update the order
        if key in self.cache:
            self.cache.pop(key)
        
        # Add to the end (most recently used)
        self.cache[key] = value
        
        # Evict least recently used item if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove first item (LRU)
```

### Redis Cache

The Redis cache provides a persistent storage layer with TTL-based expiration:

- Default TTL is 24 hours (86,400 seconds), configurable via environment variables
- Configured with `maxmemory 512mb` and `maxmemory-policy allkeys-lru` for automatic LRU eviction
- Handles both JSON data and binary data (for images)

```python
class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, password=None, ttl=86400):
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=False
        )
        self.ttl = ttl
```

### Redis Configuration

The Redis instance is configured in `docker-compose.yml` with the following settings:

```yaml
redis:
  image: redis:7-alpine
  restart: always
  ports:
    - "6379:6379"
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
  volumes:
    - redis_data:/data
```

- `maxmemory 512mb`: Limits Redis memory usage to 512 MB
- `maxmemory-policy allkeys-lru`: When memory limit is reached, Redis will remove least recently used keys

## Serialization Format

- **Main data**: JSON serialization for the response payload
- **Image data**: Stored separately in Redis using key pattern `{query}:image:{image_id}`
- **Cache keys**: The exact query string is used as the cache key

## Environment Variables

The caching system can be configured using the following environment variables:

- `MEMORY_CACHE_SIZE`: Maximum number of entries in the in-memory cache (default: 5000)
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis password (default: None)
- `REDIS_TTL`: Time-to-live for Redis cache entries in seconds (default: 86400)

## Performance Considerations

- The in-memory cache provides sub-millisecond access times for frequently accessed queries
- Redis provides persistence and larger capacity at the cost of slightly higher latency
- Using the exact query string as the cache key ensures exact matches only
- Future improvements could include semantic similarity caching for similar but not identical queries

## Monitoring and Maintenance

The API response includes a `cache_hit` field in the metadata that indicates the source of the response:
- `"memory"`: Hit in the in-memory cache
- `"redis"`: Hit in the Redis cache
- `"miss"`: Not found in either cache

This allows for monitoring cache effectiveness and hit rates. 