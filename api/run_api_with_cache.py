#!/usr/bin/env python
"""
Run the MultiModal RAG API with the two-tier caching layer.
This script starts the API server with the caching layer enabled.
"""

import os
import sys
import time
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if Redis is available
def check_redis_connection():
    """Check if Redis is available."""
    try:
        import redis
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD", None),
            socket_timeout=2
        )
        return client.ping()
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return False

def main():
    """Run the API server."""
    # Check Redis connection
    redis_available = check_redis_connection()
    if redis_available:
        print("✅ Redis is available and will be used for caching")
    else:
        print("⚠️ Redis is not available, only in-memory cache will be used")
        print("   Start Redis with: docker-compose up -d redis")
    
    # Print cache configuration
    print("\nCache Configuration:")
    print(f"In-memory cache size: {os.getenv('MEMORY_CACHE_SIZE', '5000')} entries")
    if redis_available:
        print(f"Redis TTL: {os.getenv('REDIS_TTL', '86400')} seconds (24 hours)")
        print(f"Redis host: {os.getenv('REDIS_HOST', 'localhost')}")
        print(f"Redis port: {os.getenv('REDIS_PORT', '6379')}")
        print(f"Redis DB: {os.getenv('REDIS_DB', '0')}")
    
    # Start the API server
    print("\nStarting API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 