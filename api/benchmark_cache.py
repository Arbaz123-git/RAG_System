#!/usr/bin/env python
"""
Benchmark script for the two-tier LRU cache.
This script tests the performance of the caching layer with repeated queries.
"""

import os
import sys
import time
import json
import random
import requests
import statistics
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API settings
API_URL = "http://localhost:8000"
USERNAME = "clinician1"
PASSWORD = "secret1"

# Benchmark settings
NUM_QUERIES = 10
NUM_REPEATS = 3
QUERIES = [
    "What are the characteristics of polyps in colonoscopy images?",
    "How can imaging help in identifying early-stage colon cancer?",
    "What's the difference between benign and malignant polyps?",
    "How do diverticula appear in colonoscopy images?",
    "What are the imaging features of inflammatory bowel disease?"
]

def get_token():
    """Get an authentication token from the API."""
    response = requests.post(
        f"{API_URL}/token",
        data={"username": USERNAME, "password": PASSWORD}
    )
    return response.json()["access_token"]

def benchmark_query(query, token, run_number):
    """Benchmark a single query."""
    headers = {"Authorization": f"Bearer {token}"}
    
    # Send the query to the API
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/ask",
        json={"query": query},
        headers=headers
    )
    end_time = time.time()
    
    # Calculate response time
    response_time = end_time - start_time
    
    # Get the cache hit status
    result = response.json()
    cache_hit = result["metadata"].get("cache_hit", "unknown")
    
    print(f"Run {run_number}: Query: '{query[:50]}...' - Time: {response_time:.3f}s - Cache: {cache_hit}")
    
    return {
        "query": query,
        "run": run_number,
        "time": response_time,
        "cache_hit": cache_hit
    }

def main():
    """Run the benchmark."""
    print("Starting cache benchmark...")
    
    # Get authentication token
    print("Getting authentication token...")
    token = get_token()
    
    # Results storage
    results = []
    
    # Run benchmark
    print("\nRunning benchmark...\n")
    for i in range(NUM_REPEATS):
        print(f"Repeat {i+1}/{NUM_REPEATS}")
        
        # Shuffle queries to avoid bias
        random.shuffle(QUERIES)
        
        for query in QUERIES:
            result = benchmark_query(query, token, i+1)
            results.append(result)
    
    # Analyze results
    print("\nResults Analysis:")
    
    # Group by cache hit status
    miss_times = [r["time"] for r in results if r["cache_hit"] == "miss"]
    memory_times = [r["time"] for r in results if r["cache_hit"] == "memory"]
    redis_times = [r["time"] for r in results if r["cache_hit"] == "redis"]
    
    # Calculate statistics
    print(f"\nCache misses: {len(miss_times)}")
    if miss_times:
        print(f"  Average time: {statistics.mean(miss_times):.3f}s")
        print(f"  Min time: {min(miss_times):.3f}s")
        print(f"  Max time: {max(miss_times):.3f}s")
    
    print(f"\nMemory cache hits: {len(memory_times)}")
    if memory_times:
        print(f"  Average time: {statistics.mean(memory_times):.3f}s")
        print(f"  Min time: {min(memory_times):.3f}s")
        print(f"  Max time: {max(memory_times):.3f}s")
    
    print(f"\nRedis cache hits: {len(redis_times)}")
    if redis_times:
        print(f"  Average time: {statistics.mean(redis_times):.3f}s")
        print(f"  Min time: {min(redis_times):.3f}s")
        print(f"  Max time: {max(redis_times):.3f}s")
    
    # Calculate speedup
    if miss_times and memory_times:
        memory_speedup = statistics.mean(miss_times) / statistics.mean(memory_times)
        print(f"\nMemory cache speedup: {memory_speedup:.2f}x")
    
    if miss_times and redis_times:
        redis_speedup = statistics.mean(miss_times) / statistics.mean(redis_times)
        print(f"Redis cache speedup: {redis_speedup:.2f}x")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cache_benchmark_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main() 