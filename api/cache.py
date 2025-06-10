#!/usr/bin/env python
"""
Two-Tier LRU Caching Layer for MultiModal RAG
This module implements a two-tier LRU caching system with an in-memory cache
and a Redis-based persistent cache.
"""

import json
import base64
import time
from functools import lru_cache
from collections import OrderedDict
import redis
from typing import Dict, Any, Optional, Tuple

# Default configuration
DEFAULT_IN_MEMORY_CACHE_SIZE = 5000
DEFAULT_REDIS_TTL = 86400  # 24 hours in seconds

class InMemoryLRUCache:
    """
    In-memory LRU cache implementation using OrderedDict.
    This provides a fixed-size dictionary that evicts the least recently used items.
    """
    
    def __init__(self, max_size: int = DEFAULT_IN_MEMORY_CACHE_SIZE):
        """
        Initialize the in-memory LRU cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache. If found, move it to the end (most recently used).
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            The cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        # Move to end (mark as most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Add or update an item in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # If key exists, remove it first to update the order
        if key in self.cache:
            self.cache.pop(key)
        
        # Add to the end (most recently used)
        self.cache[key] = value
        
        # Evict least recently used item if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove first item (least recently used)

class RedisCache:
    """
    Redis-based cache implementation for persistent storage.
    """
    
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 6379, 
        db: int = 0, 
        password: str = None,
        ttl: int = DEFAULT_REDIS_TTL
    ):
        """
        Initialize the Redis cache.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            ttl: Time-to-live for cache entries in seconds
        """
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Keep as bytes for binary data
        )
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get an item from Redis cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            The cached value or None if not found
        """
        try:
            # Get the main data (JSON)
            data_bytes = self.client.get(f"{key}:data")
            if not data_bytes:
                return None
            
            # Deserialize JSON data
            data = json.loads(data_bytes)
            
            # Check if there's associated image data
            image_keys = self.client.keys(f"{key}:image:*")
            if image_keys:
                data["images"] = {}
                for img_key in image_keys:
                    img_id = img_key.decode('utf-8').split(':')[-1]
                    data["images"][img_id] = self.client.get(img_key)
            
            return data
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Dict[str, Any], images: Dict[str, bytes] = None) -> bool:
        """
        Add or update an item in Redis cache.
        
        Args:
            key: Cache key
            value: Value to store (will be JSON serialized)
            images: Optional dict of image data {image_id: image_bytes}
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the main data as JSON
            self.client.setex(
                f"{key}:data",
                self.ttl,
                json.dumps(value)
            )
            
            # Store any associated image data separately
            if images:
                for img_id, img_data in images.items():
                    self.client.setex(
                        f"{key}:image:{img_id}",
                        self.ttl,
                        img_data
                    )
            
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False

class TwoTierCache:
    """
    Two-tier caching system combining in-memory LRU cache and Redis persistent cache.
    """
    
    def __init__(
        self,
        memory_cache_size: int = DEFAULT_IN_MEMORY_CACHE_SIZE,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str = None,
        redis_ttl: int = DEFAULT_REDIS_TTL
    ):
        """
        Initialize the two-tier cache.
        
        Args:
            memory_cache_size: Maximum number of items in the in-memory cache
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            redis_ttl: Time-to-live for Redis cache entries in seconds
        """
        self.memory_cache = InMemoryLRUCache(max_size=memory_cache_size)
        
        # Initialize Redis cache (with error handling)
        try:
            self.redis_cache = RedisCache(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                ttl=redis_ttl
            )
            self.redis_available = True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_available = False
    
    def get(self, key: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Get an item from the cache, checking in-memory first, then Redis.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Tuple of (cached_value, cache_source) where cache_source is one of:
            "memory", "redis", or "miss"
        """
        # Check in-memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value, "memory"
        
        # If not in memory and Redis is available, check Redis
        if self.redis_available:
            value = self.redis_cache.get(key)
            if value is not None:
                # Add to in-memory cache for faster future access
                self.memory_cache.set(key, value)
                return value, "redis"
        
        # Cache miss
        return None, "miss"
    
    def set(self, key: str, value: Dict[str, Any], images: Dict[str, bytes] = None) -> None:
        """
        Add or update an item in both cache layers.
        
        Args:
            key: Cache key
            value: Value to store
            images: Optional dict of image data {image_id: image_bytes}
        """
        # Add to in-memory cache
        self.memory_cache.set(key, value)
        
        # If Redis is available, add to Redis cache
        if self.redis_available:
            self.redis_cache.set(key, value, images) 