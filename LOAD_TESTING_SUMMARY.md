# Two-Tier LRU Caching with Load Testing Implementation

## Overview

We've implemented a comprehensive two-tier LRU caching system for the MultiModal RAG API, along with load testing capabilities to verify performance under high concurrency.

## Components Implemented

### 1. Two-Tier Caching System
- **In-Memory LRU Cache**: Fast, fixed-size dictionary using `OrderedDict` with LRU eviction policy
- **Redis Cache**: Persistent layer with TTL-based expiration and LRU eviction when memory limits are reached
- **Cache Integration**: Seamlessly integrated with the existing API to check both cache layers before performing expensive RAG operations

### 2. Load Testing Framework
- **Locust Scripts**: Python-based load testing using Locust to simulate 100 concurrent users
- **Resource Monitoring**: Tracking of CPU and memory usage during tests
- **Cache Performance Metrics**: Tracking of cache hit rates and response times by cache type
- **Automated Testing**: Script to run headless tests and generate reports

### 3. Performance Monitoring
- **Cache Statistics API**: Endpoint to view current cache statistics
- **Latency Tracking**: Measurement of response times for different cache scenarios
- **Report Generation**: Automatic generation of Markdown reports with performance metrics

## Key Files

- `api/cache.py`: Implementation of the two-tier LRU cache
- `api/cache_monitor.py`: Cache statistics tracking and reporting
- `api/main.py`: API with integrated caching and monitoring
- `loadtest.py`: Locust load testing script
- `run_loadtest.py`: Script to run automated load tests
- `reports/loadtest.md`: Load test report template
- `reports/README.md`: Documentation for running load tests
- `cache_engineering_dossier.md`: Technical documentation of the caching system

## How to Run

1. Start the API with caching:
   ```bash
   cd api
   python run_api_with_cache.py
   ```

2. Run the load test:
   ```bash
   python run_loadtest.py
   ```

3. View the results in the `reports` directory

## Performance Target

The system is designed to meet the requirement of **p95 < 250ms at 100 req/s**, meaning that 95% of requests should complete in less than 250ms under a load of 100 requests per second.

The two-tier caching system significantly improves response times:
- Cache misses: Full RAG pipeline execution (slowest)
- Redis cache hits: Faster than misses, requires network I/O
- Memory cache hits: Fastest responses, sub-millisecond latency

## Redis Configuration

Redis is configured with:
- `maxmemory 512mb`: Limits Redis memory usage to 512 MB
- `maxmemory-policy allkeys-lru`: Automatically evicts least recently used keys when memory limit is reached

## Future Improvements

1. Semantic similarity caching for similar but not identical queries
2. Cache warming for common queries
3. More granular cache expiration policies based on content type
4. Further optimization of Redis connection pooling for higher throughput 