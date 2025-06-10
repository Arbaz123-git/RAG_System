#!/usr/bin/env python
"""
Cache monitoring script for the MultiModal RAG API.
This script tracks and saves cache statistics during load testing.
"""

import os
import json
import time
import threading
from datetime import datetime

# Default path for saving statistics
DEFAULT_STATS_PATH = "../reports/cache_stats.json"

class CacheMonitor:
    """Monitor and record cache statistics."""
    
    def __init__(self, stats_path=DEFAULT_STATS_PATH):
        """Initialize the cache monitor."""
        self.stats = {
            "miss": 0,
            "memory": 0,
            "redis": 0,
            "unknown": 0,
            "timestamps": [],
            "latencies": {
                "miss": [],
                "memory": [],
                "redis": []
            }
        }
        self.stats_path = stats_path
        self.lock = threading.Lock()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    
    def record_cache_access(self, cache_type, latency_ms):
        """Record a cache access with its latency."""
        with self.lock:
            if cache_type in self.stats:
                self.stats[cache_type] += 1
            else:
                self.stats["unknown"] += 1
            
            # Record timestamp
            self.stats["timestamps"].append({
                "time": datetime.now().isoformat(),
                "type": cache_type
            })
            
            # Record latency if it's a known type
            if cache_type in self.stats["latencies"]:
                self.stats["latencies"][cache_type].append(latency_ms)
    
    def save_stats(self):
        """Save the current statistics to a file."""
        with self.lock:
            # Calculate summary statistics for latencies
            summary = {
                "total_requests": sum(v for k, v in self.stats.items() 
                                     if k not in ["timestamps", "latencies"]),
                "cache_hits": self.stats["memory"] + self.stats["redis"],
                "cache_misses": self.stats["miss"],
                "unknown": self.stats["unknown"],
                "hit_rate": 0,
                "avg_latencies": {}
            }
            
            if summary["total_requests"] > 0:
                summary["hit_rate"] = (summary["cache_hits"] / summary["total_requests"]) * 100
            
            # Calculate average latencies
            for cache_type, latencies in self.stats["latencies"].items():
                if latencies:
                    summary["avg_latencies"][cache_type] = sum(latencies) / len(latencies)
                else:
                    summary["avg_latencies"][cache_type] = 0
            
            # Combine stats and summary
            output = {
                "stats": self.stats,
                "summary": summary
            }
            
            # Save to file
            with open(self.stats_path, "w") as f:
                json.dump(output, f, indent=2)
    
    def start_periodic_save(self, interval=30):
        """Start a thread to periodically save statistics."""
        def save_loop():
            while True:
                time.sleep(interval)
                self.save_stats()
        
        thread = threading.Thread(target=save_loop, daemon=True)
        thread.start()
        return thread

# Global instance for use in the API
cache_monitor = CacheMonitor()

# Start periodic saving when imported
save_thread = cache_monitor.start_periodic_save()

if __name__ == "__main__":
    # For testing
    print("Cache monitor started. Press Ctrl+C to stop.")
    try:
        # Simulate some cache accesses
        for i in range(10):
            cache_monitor.record_cache_access("miss", 500)
            time.sleep(0.1)
            cache_monitor.record_cache_access("memory", 10)
            time.sleep(0.1)
            cache_monitor.record_cache_access("redis", 50)
            time.sleep(0.1)
        
        # Wait for a while
        time.sleep(5)
        
        # Save and exit
        cache_monitor.save_stats()
        print(f"Statistics saved to {cache_monitor.stats_path}")
    
    except KeyboardInterrupt:
        print("Saving statistics and exiting...")
        cache_monitor.save_stats()
        print(f"Statistics saved to {cache_monitor.stats_path}") 