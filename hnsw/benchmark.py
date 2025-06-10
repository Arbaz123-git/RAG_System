import numpy as np
import time
import argparse
import os
import pickle
from typing import List, Dict, Any, Tuple

from hnsw import HNSW
from utils import (
    generate_random_vectors,
    compute_exact_neighbors,
    evaluate_recall,
    benchmark_index_construction,
    benchmark_query_performance,
    plot_recall_vs_qps,
    plot_construction_time_vs_params,
    analyze_complexity
)

def create_hnsw_index(data: np.ndarray, **params) -> HNSW:
    """
    Create and populate HNSW index
    
    Parameters:
    -----------
    data: np.ndarray
        Data vectors
    params: Dict
        Parameters for HNSW constructor
        
    Returns:
    --------
    HNSW: Populated index
    """
    dim = data.shape[1]
    max_elements = data.shape[0]
    
    # Create index
    index = HNSW(
        dim=dim,
        max_elements=max_elements,
        ef_construction=params.get('ef_construction', 200),
        M=params.get('M', 16),
        ef_search=params.get('ef_search', 50),
        distance_func=params.get('distance_func', 'l2')
    )
    
    # Add vectors to index
    for i in range(data.shape[0]):
        index.add(data[i], i)
        
        # Print progress
        if (i + 1) % 10000 == 0:
            print(f"Added {i + 1}/{data.shape[0]} vectors to index")
            
    return index

def query_hnsw_index(index: HNSW, query: np.ndarray, k: int) -> np.ndarray:
    """
    Query HNSW index
    
    Parameters:
    -----------
    index: HNSW
        Index to query
    query: np.ndarray
        Query vector
    k: int
        Number of neighbors to find
        
    Returns:
    --------
    np.ndarray: Indices of nearest neighbors
    """
    results = index.search(query, k)
    return np.array([r[1] for r in results])

def run_parameter_sweep(
    data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> Dict[str, Any]:
    """
    Run parameter sweep to find optimal parameters
    
    Parameters:
    -----------
    data: np.ndarray
        Data vectors
    queries: np.ndarray
        Query vectors
    ground_truth: np.ndarray
        Ground truth indices
    k: int
        Number of neighbors to find
        
    Returns:
    --------
    Dict[str, Any]: Benchmark results
    """
    # Parameters to sweep
    m_values = [4, 8, 16, 32, 64]
    ef_construction_values = [40, 80, 160, 320, 640]
    ef_search_values = [20, 40, 80, 160, 320]
    
    # Results storage
    construction_results = []
    query_results = []
    
    # Test different M values
    for M in m_values:
        params = {
            'M': M,
            'ef_construction': 200,
            'ef_search': 100,
            'distance_func': 'l2'
        }
        
        print(f"Testing M={M}")
        index, construction_time = benchmark_index_construction(
            create_hnsw_index, data, params
        )
        
        construction_results.append({
            'params': params,
            'construction_time': construction_time
        })
        
        # Benchmark query performance
        perf = benchmark_query_performance(
            index, query_hnsw_index, queries, k, ground_truth
        )
        perf['label'] = f"M={M}"
        perf['params'] = params
        query_results.append(perf)
        
        print(f"  Recall: {perf['recall']:.4f}, QPS: {perf['qps']:.2f}")
        
    # Test different ef_construction values
    for ef_construction in ef_construction_values:
        params = {
            'M': 16,
            'ef_construction': ef_construction,
            'ef_search': 100,
            'distance_func': 'l2'
        }
        
        print(f"Testing ef_construction={ef_construction}")
        index, construction_time = benchmark_index_construction(
            create_hnsw_index, data, params
        )
        
        construction_results.append({
            'params': params,
            'construction_time': construction_time
        })
        
        # Benchmark query performance
        perf = benchmark_query_performance(
            index, query_hnsw_index, queries, k, ground_truth
        )
        perf['label'] = f"ef_c={ef_construction}"
        perf['params'] = params
        query_results.append(perf)
        
        print(f"  Recall: {perf['recall']:.4f}, QPS: {perf['qps']:.2f}")
        
    # Test different ef_search values
    best_index = None
    best_recall = 0
    
    for ef_search in ef_search_values:
        params = {
            'M': 16,
            'ef_construction': 200,
            'ef_search': ef_search,
            'distance_func': 'l2'
        }
        
        print(f"Testing ef_search={ef_search}")
        if best_index is None:
            index, construction_time = benchmark_index_construction(
                create_hnsw_index, data, params
            )
            best_index = index
        else:
            # Reuse the same index but change ef_search
            index = best_index
            index.ef_search = ef_search
            construction_time = 0
        
        # Benchmark query performance
        perf = benchmark_query_performance(
            index, query_hnsw_index, queries, k, ground_truth
        )
        perf['label'] = f"ef_s={ef_search}"
        perf['params'] = params
        query_results.append(perf)
        
        print(f"  Recall: {perf['recall']:.4f}, QPS: {perf['qps']:.2f}")
        
        # Keep track of best recall
        if perf['recall'] > best_recall:
            best_recall = perf['recall']
            
    # Plot results
    plot_construction_time_vs_params(
        construction_results[:len(m_values)], 'M', 'Construction Time vs. M'
    )
    plot_construction_time_vs_params(
        construction_results[len(m_values):], 'ef_construction', 'Construction Time vs. ef_construction'
    )
    plot_recall_vs_qps(query_results[:len(m_values)], 'Recall vs. QPS (M)')
    plot_recall_vs_qps(query_results[len(m_values):len(m_values)+len(ef_construction_values)], 
                      'Recall vs. QPS (ef_construction)')
    plot_recall_vs_qps(query_results[len(m_values)+len(ef_construction_values):], 
                      'Recall vs. QPS (ef_search)')
    
    return {
        'construction_results': construction_results,
        'query_results': query_results
    }

def analyze_search_complexity(
    data: np.ndarray,
    queries: np.ndarray,
    k: int
) -> Dict[str, Any]:
    """
    Analyze search complexity
    
    Parameters:
    -----------
    data: np.ndarray
        Data vectors
    queries: np.ndarray
        Query vectors
    k: int
        Number of neighbors to find
        
    Returns:
    --------
    Dict[str, Any]: Complexity analysis results
    """
    # Create full index
    params = {
        'M': 16,
        'ef_construction': 200,
        'ef_search': 100,
        'distance_func': 'l2'
    }
    
    print("Creating index for complexity analysis...")
    index = create_hnsw_index(data, **params)
    
    # Dataset sizes to test
    dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    dataset_sizes = [s for s in dataset_sizes if s <= data.shape[0]]
    
    print("Analyzing search complexity...")
    complexity_results = analyze_complexity(
        index, queries[:100], query_hnsw_index, k, dataset_sizes
    )
    
    return complexity_results

def main():
    parser = argparse.ArgumentParser(description='Benchmark HNSW implementation')
    parser.add_argument('--data_size', type=int, default=100000,
                        help='Number of data vectors')
    parser.add_argument('--query_size', type=int, default=1000,
                        help='Number of query vectors')
    parser.add_argument('--dim', type=int, default=128,
                        help='Vector dimensionality')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'sweep', 'complexity', 'recall'],
                        help='Benchmark mode')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate data
    print(f"Generating {args.data_size} data vectors...")
    data = generate_random_vectors(args.data_size, args.dim, args.seed)
    
    print(f"Generating {args.query_size} query vectors...")
    queries = generate_random_vectors(args.query_size, args.dim, args.seed + 1)
    
    # Compute ground truth
    print("Computing exact nearest neighbors...")
    ground_truth = compute_exact_neighbors(data, queries, args.k)
    
    # Run benchmarks
    results = {}
    
    if args.mode in ['all', 'sweep']:
        print("Running parameter sweep...")
        sweep_results = run_parameter_sweep(data, queries, ground_truth, args.k)
        results['sweep'] = sweep_results
        
    if args.mode in ['all', 'complexity']:
        print("Analyzing search complexity...")
        complexity_results = analyze_search_complexity(data, queries, args.k)
        results['complexity'] = complexity_results
        
    if args.mode in ['all', 'recall']:
        # Find parameters that achieve Recall@10 ≥ 0.92
        print("Finding parameters for Recall@10 ≥ 0.92...")
        
        # Try different parameter combinations
        best_params = None
        best_recall = 0
        best_qps = 0
        
        for M in [16, 24, 32]:
            for ef_construction in [100, 200, 300]:
                for ef_search in [50, 100, 200]:
                    params = {
                        'M': M,
                        'ef_construction': ef_construction,
                        'ef_search': ef_search,
                        'distance_func': 'l2'
                    }
                    
                    print(f"Testing params: {params}")
                    index, _ = benchmark_index_construction(
                        create_hnsw_index, data, params
                    )
                    
                    # Test on a larger query set
                    test_queries = queries[:5000] if queries.shape[0] >= 5000 else queries
                    test_ground_truth = ground_truth[:5000] if ground_truth.shape[0] >= 5000 else ground_truth
                    
                    perf = benchmark_query_performance(
                        index, query_hnsw_index, test_queries, args.k, test_ground_truth
                    )
                    
                    print(f"  Recall: {perf['recall']:.4f}, QPS: {perf['qps']:.2f}")
                    
                    # Check if recall meets the target
                    if perf['recall'] >= 0.92:
                        if best_params is None or perf['qps'] > best_qps:
                            best_params = params
                            best_recall = perf['recall']
                            best_qps = perf['qps']
        
        if best_params is not None:
            print(f"Found parameters achieving Recall@10 ≥ 0.92:")
            print(f"  Parameters: {best_params}")
            print(f"  Recall: {best_recall:.4f}")
            print(f"  QPS: {best_qps:.2f}")
            
            # Create index with best parameters
            best_index, _ = benchmark_index_construction(
                create_hnsw_index, data, best_params
            )
            
            # Save the best index
            best_index.save(os.path.join(args.save_dir, 'best_hnsw_index.npy'))
            
            results['best_params'] = {
                'params': best_params,
                'recall': best_recall,
                'qps': best_qps
            }
        else:
            print("Could not find parameters achieving Recall@10 ≥ 0.92")
    
    # Save results
    with open(os.path.join(args.save_dir, 'benchmark_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("Benchmark completed!")

if __name__ == "__main__":
    main() 