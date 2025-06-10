import numpy as np
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
import os

from hnsw import HNSW
from utils import generate_random_vectors, compute_exact_neighbors, evaluate_recall

def main():
    # Parameters
    dim = 128
    n_data = 50000
    n_queries = 1000
    k = 10
    
    print(f"Generating {n_data} random vectors...")
    data = generate_random_vectors(n_data, dim, seed=42)
    
    print(f"Generating {n_queries} query vectors...")
    queries = generate_random_vectors(n_queries, dim, seed=43)
    
    # Compute ground truth for evaluation
    print("Computing exact nearest neighbors (ground truth)...")
    start_time = time.time()
    ground_truth = compute_exact_neighbors(data, queries, k)
    exact_time = time.time() - start_time
    print(f"Exact search took {exact_time:.2f} seconds ({n_queries / exact_time:.2f} QPS)")
    
    # Create and populate HNSW index
    print("\nCreating HNSW index...")
    index_params = {
        'M': 16,
        'ef_construction': 200,
        'ef_search': 100
    }
    
    index = HNSW(
        dim=dim,
        max_elements=n_data,
        ef_construction=index_params['ef_construction'],
        M=index_params['M'],
        ef_search=index_params['ef_search']
    )
    
    # Add vectors to index
    start_time = time.time()
    for i in range(n_data):
        index.add(data[i], i)
        if (i + 1) % 10000 == 0:
            print(f"Added {i + 1}/{n_data} vectors to index")
            
    index_time = time.time() - start_time
    print(f"Index construction took {index_time:.2f} seconds ({n_data / index_time:.2f} vectors/second)")
    
    # Print index statistics
    stats = index.get_stats()
    print("\nIndex statistics:")
    print(f"Number of elements: {stats['num_elements']}")
    print(f"Maximum level: {stats['max_level']}")
    
    for level, level_stats in stats['layers'].items():
        print(f"Level {level}: {level_stats['nodes']} nodes, "
              f"{level_stats['avg_connections']:.2f} avg connections")
    
    # Search using HNSW
    print("\nSearching with HNSW...")
    approximate_results = []
    
    start_time = time.time()
    for i in range(n_queries):
        results = index.search(queries[i], k)
        approximate_results.append([r[1] for r in results])
        
    hnsw_time = time.time() - start_time
    print(f"HNSW search took {hnsw_time:.2f} seconds ({n_queries / hnsw_time:.2f} QPS)")
    print(f"Speedup over exact search: {exact_time / hnsw_time:.2f}x")
    
    # Process approximate results to ensure consistent shape
    processed_results = []
    for result in approximate_results:
        # Ensure result is a numpy array
        if not isinstance(result, np.ndarray):
            result = np.array(result)
            
        # Ensure result has length k by padding or truncating
        if len(result) < k:
            # Pad with -1 if too short
            padded = np.full(k, -1)
            padded[:len(result)] = result
            processed_results.append(padded)
        else:
            # Truncate if too long
            processed_results.append(result[:k])
            
    # Convert to numpy array
    approximate_results = np.array(processed_results)
    
    # Evaluate recall
    recall = evaluate_recall(ground_truth, approximate_results)
    print(f"Recall@{k}: {recall:.4f}")
    
    # Try different ef_search values to find the best trade-off
    print("\nTesting different ef_search values...")
    ef_values = [20, 50, 100, 200, 500]
    recalls = []
    qps = []
    
    for ef in ef_values:
        index.ef_search = ef
        
        start_time = time.time()
        results = []
        
        for i in range(n_queries):
            query_results = index.search(queries[i], k)
            results.append([r[1] for r in query_results])
            
        query_time = time.time() - start_time
        
        # Process results to ensure consistent shape
        processed_results = []
        for result in results:
            # Ensure result is a numpy array
            if not isinstance(result, np.ndarray):
                result = np.array(result)
                
            # Ensure result has length k by padding or truncating
            if len(result) < k:
                # Pad with -1 if too short
                padded = np.full(k, -1)
                padded[:len(result)] = result
                processed_results.append(padded)
            else:
                # Truncate if too long
                processed_results.append(result[:k])
                
        # Convert to numpy array
        results = np.array(processed_results)
        
        # Calculate recall
        recall = evaluate_recall(ground_truth, results)
        
        recalls.append(recall)
        qps.append(n_queries / query_time)
        
        print(f"ef_search={ef}: Recall@{k}={recall:.4f}, QPS={n_queries / query_time:.2f}")
    
    # Plot recall vs QPS
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, qps, 'o-', linewidth=2, markersize=10)
    
    for i, ef in enumerate(ef_values):
        plt.annotate(f"ef={ef}", (recalls[i], qps[i]), fontsize=12)
        
    plt.xlabel("Recall@10", fontsize=14)
    plt.ylabel("Queries per second", fontsize=14)
    plt.title("Recall vs. QPS trade-off", fontsize=16)
    plt.grid(True)
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/recall_vs_qps.png")
    
    print("\nPlot saved to results/recall_vs_qps.png")
    
    # Save the index
    index.save("results/hnsw_index.npy")
    print("Index saved to results/hnsw_index.npy")
    
    # Demonstrate search complexity is O(log N)
    print("\nDemonstrating O(log N) search complexity...")
    
    # Create a new index with more data
    large_n = 200000
    print(f"Generating {large_n} random vectors...")
    large_data = generate_random_vectors(large_n, dim, seed=44)
    
    large_index = HNSW(
        dim=dim,
        max_elements=large_n,
        ef_construction=200,
        M=16,
        ef_search=100
    )
    
    # Add vectors to index
    print("Building index with 200,000 vectors...")
    for i in range(large_n):
        large_index.add(large_data[i], i)
        if (i + 1) % 50000 == 0:
            print(f"Added {i + 1}/{large_n} vectors to index")
    
    # Test search times for different dataset sizes
    sizes = [1000, 10000, 50000, 100000, 200000]
    times = []
    
    test_queries = generate_random_vectors(100, dim, seed=45)
    
    for size in sizes:
        print(f"Testing search time for {size} vectors...")
        
        # Measure search time
        start_time = time.time()
        for i in range(100):
            large_index.search(test_queries[i % 100], k)
        query_time = (time.time() - start_time) / 100
        
        times.append(query_time)
        print(f"Average search time: {query_time * 1000:.2f} ms")
    
    # Plot search time vs dataset size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=10)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel("Dataset Size (N)", fontsize=14)
    plt.ylabel("Search Time (s)", fontsize=14)
    plt.title("Search Time vs. Dataset Size", fontsize=16)
    plt.grid(True)
    
    # Fit log curve to confirm O(log N) complexity
    log_sizes = np.log(sizes)
    coeffs = np.polyfit(log_sizes, times, 1)
    
    # Plot the fitted curve
    fit_times = coeffs[0] * log_sizes + coeffs[1]
    plt.plot(sizes, np.exp(fit_times), 'r--', linewidth=2, 
             label=f'Fit: {coeffs[0]:.4f} * log(N) + {coeffs[1]:.4f}')
    
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.savefig("results/search_complexity.png")
    print("Search complexity plot saved to results/search_complexity.png")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main() 