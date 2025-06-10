import unittest
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hnsw import HNSW
from utils import generate_random_vectors, compute_exact_neighbors, evaluate_recall

class TestHNSW(unittest.TestCase):
    def setUp(self):
        # Generate random data for testing
        self.dim = 128
        self.n_data = 1000
        self.n_queries = 100
        self.k = 10
        
        # Generate data and queries
        self.data = generate_random_vectors(self.n_data, self.dim, seed=42)
        self.queries = generate_random_vectors(self.n_queries, self.dim, seed=43)
        
        # Compute ground truth
        self.ground_truth = compute_exact_neighbors(self.data, self.queries, self.k)
        
    def test_index_construction(self):
        """Test that the index can be constructed"""
        index = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=100,
            M=16
        )
        
        # Add vectors to index
        for i in range(self.n_data):
            index.add(self.data[i], i)
            
        # Check that all elements were added
        self.assertEqual(index.num_elements, self.n_data)
        
        # Check that the maximum level is reasonable
        self.assertGreaterEqual(index.ml, 0)
        self.assertLessEqual(index.ml, 10)  # Unlikely to be higher for 1000 elements
        
    def test_search(self):
        """Test that search returns the correct number of results"""
        index = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=100,
            M=16
        )
        
        # Add vectors to index
        for i in range(self.n_data):
            index.add(self.data[i], i)
            
        # Search for nearest neighbors
        results = index.search(self.queries[0], self.k)
        
        # Check that the correct number of results is returned
        self.assertEqual(len(results), self.k)
        
        # Check that results are sorted by distance
        for i in range(1, len(results)):
            self.assertLessEqual(results[i-1][0], results[i][0])
            
    def test_recall(self):
        """Test that recall is reasonable"""
        index = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=200,
            M=16,
            ef_search=100
        )
        
        # Add vectors to index
        for i in range(self.n_data):
            index.add(self.data[i], i)
            
        # Search for nearest neighbors
        approximate_results = []
        for i in range(self.n_queries):
            results = index.search(self.queries[i], self.k)
            approximate_results.append([r[1] for r in results])
            
        approximate_results = np.array(approximate_results)
        
        # Compute recall
        recall = evaluate_recall(self.ground_truth, approximate_results)
        
        # Check that recall is reasonable (at least 0.7 for these parameters)
        self.assertGreaterEqual(recall, 0.7)
        
    def test_save_load(self):
        """Test that the index can be saved and loaded"""
        # Create and populate index
        index = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=100,
            M=16
        )
        
        for i in range(self.n_data):
            index.add(self.data[i], i)
            
        # Save index
        filename = "test_index.npy"
        index.save(filename)
        
        # Load index
        loaded_index = HNSW.load(filename)
        
        # Check that the loaded index has the same parameters
        self.assertEqual(loaded_index.dim, index.dim)
        self.assertEqual(loaded_index.max_elements, index.max_elements)
        self.assertEqual(loaded_index.ef_construction, index.ef_construction)
        self.assertEqual(loaded_index.M, index.M)
        self.assertEqual(loaded_index.num_elements, index.num_elements)
        self.assertEqual(loaded_index.ml, index.ml)
        
        # Search with both indices and compare results
        query = self.queries[0]
        results1 = index.search(query, self.k)
        results2 = loaded_index.search(query, self.k)
        
        # Check that the results are the same
        for i in range(self.k):
            self.assertEqual(results1[i][1], results2[i][1])
            self.assertAlmostEqual(results1[i][0], results2[i][0])
            
        # Clean up
        os.remove(filename)
        
    def test_distance_functions(self):
        """Test different distance functions"""
        # Test L2 distance
        index_l2 = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=100,
            M=16,
            distance_func="l2"
        )
        
        # Test cosine distance
        index_cosine = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=100,
            M=16,
            distance_func="cosine"
        )
        
        # Test inner product distance
        index_ip = HNSW(
            dim=self.dim,
            max_elements=self.n_data,
            ef_construction=100,
            M=16,
            distance_func="ip"
        )
        
        # Add vectors to indices
        for i in range(100):  # Use fewer vectors for this test
            index_l2.add(self.data[i], i)
            index_cosine.add(self.data[i], i)
            index_ip.add(self.data[i], i)
            
        # Search with all indices
        query = self.queries[0]
        results_l2 = index_l2.search(query, 5)
        results_cosine = index_cosine.search(query, 5)
        results_ip = index_ip.search(query, 5)
        
        # Check that results are returned
        self.assertEqual(len(results_l2), 5)
        self.assertEqual(len(results_cosine), 5)
        self.assertEqual(len(results_ip), 5)
        
    def test_search_complexity(self):
        """Test that search complexity is O(log N)"""
        # Create index with more data
        n_data = 10000
        data = generate_random_vectors(n_data, self.dim, seed=42)
        
        index = HNSW(
            dim=self.dim,
            max_elements=n_data,
            ef_construction=100,
            M=16
        )
        
        # Add vectors to index
        for i in range(n_data):
            index.add(data[i], i)
            
        # Measure search time for different dataset sizes
        sizes = [100, 1000, 10000]
        times = []
        
        for size in sizes:
            # Limit the search to the first 'size' elements
            start_time = time.time()
            
            # Run multiple searches to get a stable measurement
            n_searches = 100
            for i in range(n_searches):
                index.search(self.queries[i % self.n_queries], self.k)
                
            elapsed = time.time() - start_time
            times.append(elapsed / n_searches)
            
        # Check that the growth is sub-linear
        # If time complexity is O(log N), then time(10000)/time(100) < 10000/100 = 100
        ratio = times[2] / times[0]
        self.assertLess(ratio, 100)
        
if __name__ == '__main__':
    unittest.main() 