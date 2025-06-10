# MultiModal RAG Load Testing

This directory contains load testing scripts and results for the MultiModal RAG API with two-tier caching.

## Prerequisites

1. Python 3.8+ with pip
2. Locust installed (`pip install locust`)
3. Redis running (optional, for two-tier caching)
4. MultiModal RAG API running

## Running the Load Test

### Option 1: Using the UI

1. Start the MultiModal RAG API:
```bash
cd api
python run_api_with_cache.py
```

2. Start Locust with the load test script:
```bash
locust -f loadtest.py --host=http://localhost:8000
```

3. Open the Locust web interface at http://localhost:8089
4. Configure the test with:
   - Number of users: 100
   - Spawn rate: 10 users/second
   - Host: http://localhost:8000
5. Start the test and monitor results in real-time

### Option 2: Using the Automated Script

1. Start the MultiModal RAG API:
```bash
cd api
python run_api_with_cache.py
```

2. Run the automated load test script:
```bash
python run_loadtest.py
```

3. View the results in the `reports` directory:
   - `loadtest.md`: Summary report
   - `resource_monitoring.json`: CPU and memory usage data
   - `cache_stats.json`: Cache performance statistics
   - `locust_results_*.csv`: Raw test data from Locust

## Interpreting Results

The load test report (`loadtest.md`) includes:

1. **Latency Percentiles**: p50, p90, p95, and p99 response times in milliseconds
2. **Throughput**: Average requests per second and error rate
3. **Cache Performance**: Hit rates for memory cache and Redis cache
4. **Resource Usage**: CPU and memory consumption during the test

The primary success criterion is **p95 < 250ms**, meaning 95% of requests should complete in less than 250ms.

## Customizing the Test

To modify the test parameters:

1. Edit `loadtest.py` to change:
   - Query patterns
   - User behavior
   - Request distribution

2. Edit `run_loadtest.py` to change:
   - Test duration
   - Number of users
   - Spawn rate

## Troubleshooting

- **High error rates**: Check if the API is running and accessible
- **Low throughput**: Check for bottlenecks in the API or system resources
- **Redis errors**: Verify Redis is running (`docker-compose up -d redis`)
- **Memory issues**: Reduce the in-memory cache size via environment variables 