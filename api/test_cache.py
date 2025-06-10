#!/usr/bin/env python
"""
Test script for the two-tier LRU cache implementation.
This script tests both the in-memory cache and Redis cache functionality.
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the cache implementation
from api.cache import InMemoryLRUCache, RedisCache, TwoTierCache

class TestInMemoryLRUCache(unittest.TestCase):
    """Test the in-memory LRU cache implementation."""
    
    def setUp(self):
        """Set up a cache instance for testing."""
        self.cache = InMemoryLRUCache(max_size=3)
    
    def test_get_set(self):
        """Test basic get and set operations."""
        # Set some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Get values
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertIsNone(self.cache.get("key3"))
    
    def test_lru_eviction(self):
        """Test that least recently used items are evicted when the cache is full."""
        # Fill the cache
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Cache is now full, add another item
        self.cache.set("key4", "value4")
        
        # key1 should have been evicted
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")
    
    def test_lru_update(self):
        """Test that accessing an item updates its LRU status."""
        # Fill the cache
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Access key1, making key2 the least recently used
        self.cache.get("key1")
        self.cache.get("key3")
        
        # Add a new item, which should evict key2
        self.cache.set("key4", "value4")
        
        # key2 should have been evicted
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")

class TestRedisCacheMock(unittest.TestCase):
    """Test the Redis cache implementation with mocks."""
    
    def setUp(self):
        """Set up a Redis cache with a mocked Redis client."""
        # Create a mock Redis client
        self.mock_redis = MagicMock()
        
        # Patch the Redis class to return our mock
        self.redis_patcher = patch('api.cache.redis.Redis', return_value=self.mock_redis)
        self.redis_patcher.start()
        
        # Create the Redis cache
        self.cache = RedisCache(ttl=3600)
    
    def tearDown(self):
        """Clean up the patch."""
        self.redis_patcher.stop()
    
    def test_get(self):
        """Test getting a value from Redis."""
        # Setup the mock to return a value
        self.mock_redis.get.return_value = json.dumps({"answer": "test"}).encode('utf-8')
        self.mock_redis.keys.return_value = []
        
        # Get the value
        result = self.cache.get("test_key")
        
        # Check that the mock was called correctly
        self.mock_redis.get.assert_called_once_with("test_key:data")
        
        # Check the result
        self.assertEqual(result, {"answer": "test"})
    
    def test_set(self):
        """Test setting a value in Redis."""
        # Set a value
        value = {"answer": "test"}
        self.cache.set("test_key", value)
        
        # Check that the mock was called correctly
        self.mock_redis.setex.assert_called_once_with(
            "test_key:data",
            3600,
            json.dumps(value)
        )

class TestTwoTierCacheMock(unittest.TestCase):
    """Test the two-tier cache implementation with mocks."""
    
    def setUp(self):
        """Set up a two-tier cache with mocked components."""
        # Create mock caches
        self.mock_memory_cache = MagicMock()
        self.mock_redis_cache = MagicMock()
        
        # Create a two-tier cache with the mocks
        self.cache = TwoTierCache()
        self.cache.memory_cache = self.mock_memory_cache
        self.cache.redis_cache = self.mock_redis_cache
        self.cache.redis_available = True
    
    def test_get_memory_hit(self):
        """Test getting a value that's in the memory cache."""
        # Setup the memory cache to return a value
        self.mock_memory_cache.get.return_value = {"answer": "memory"}
        
        # Get the value
        result, source = self.cache.get("test_key")
        
        # Check that only the memory cache was checked
        self.mock_memory_cache.get.assert_called_once_with("test_key")
        self.mock_redis_cache.get.assert_not_called()
        
        # Check the result
        self.assertEqual(result, {"answer": "memory"})
        self.assertEqual(source, "memory")
    
    def test_get_redis_hit(self):
        """Test getting a value that's in the Redis cache but not memory."""
        # Setup the memory cache to miss and Redis to hit
        self.mock_memory_cache.get.return_value = None
        self.mock_redis_cache.get.return_value = {"answer": "redis"}
        
        # Get the value
        result, source = self.cache.get("test_key")
        
        # Check that both caches were checked
        self.mock_memory_cache.get.assert_called_once_with("test_key")
        self.mock_redis_cache.get.assert_called_once_with("test_key")
        
        # Check that the value was added to the memory cache
        self.mock_memory_cache.set.assert_called_once_with("test_key", {"answer": "redis"})
        
        # Check the result
        self.assertEqual(result, {"answer": "redis"})
        self.assertEqual(source, "redis")
    
    def test_get_miss(self):
        """Test getting a value that's not in either cache."""
        # Setup both caches to miss
        self.mock_memory_cache.get.return_value = None
        self.mock_redis_cache.get.return_value = None
        
        # Get the value
        result, source = self.cache.get("test_key")
        
        # Check that both caches were checked
        self.mock_memory_cache.get.assert_called_once_with("test_key")
        self.mock_redis_cache.get.assert_called_once_with("test_key")
        
        # Check the result
        self.assertIsNone(result)
        self.assertEqual(source, "miss")
    
    def test_set(self):
        """Test setting a value in both caches."""
        # Set a value
        value = {"answer": "test"}
        self.cache.set("test_key", value)
        
        # Check that both caches were updated
        self.mock_memory_cache.set.assert_called_once_with("test_key", value)
        self.mock_redis_cache.set.assert_called_once_with("test_key", value, None)

if __name__ == "__main__":
    unittest.main() 