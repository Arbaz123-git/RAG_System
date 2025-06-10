import numpy as np
import time
from typing import List, Tuple, Dict, Any, Callable
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

def generate_random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """
    Generate random vectors for testing
    
    Parameters:
    -----------
    n: int
        Number of vectors
    dim: int
        Dimensionality of vectors
    seed: int
        Random seed
        
    Returns:
    --------
    np.ndarray: Random vectors of shape (n, dim)
    """
    np.random.seed(seed)
    return np.random.random((n, dim)).astype(np.float32)

def compute_exact_neighbors(data: np.ndarray, queries: np.ndarray, k: int, metric: str = 'l2') -> np.ndarray:
    """
    Compute exact nearest neighbors using sklearn
    
    Parameters:
    -----------
    data: np.ndarray
        Data vectors of shape (n, dim)
    queries: np.ndarray
        Query vectors of shape (m, dim)
    k: int
        Number of neighbors to find
    metric: str
        Distance metric ('l2', 'cosine', 'euclidean')
        
    Returns:
    --------
    np.ndarray: Indices of nearest neighbors of shape (m, k)
    """
    if metric == 'l2':
        metric = 'euclidean'
        
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric)
    nn.fit(data)
    _, indices = nn.kneighbors(queries)
    return indices

def evaluate_recall(ground_truth: np.ndarray, approximate: np.ndarray) -> float:
    """
    Evaluate recall@k
    
    Parameters:
    -----------
    ground_truth: np.ndarray
        Ground truth indices of shape (n_queries, k)
    approximate: np.ndarray
        Approximate indices of shape (n_queries, k)
        
    Returns:
    --------
    float: Recall@k
    """
    n_queries, k = ground_truth.shape
    recalls = []
    
    for i in range(n_queries):
        gt_set = set(ground_truth[i])
        ap_set = set(approximate[i])
        recall = len(gt_set.intersection(ap_set)) / k
        recalls.append(recall)
        
    return np.mean(recalls)

def benchmark_index_construction(
    index_constructor: Callable,
    data: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[Any, float]:
    """
    Benchmark index construction time
    
    Parameters:
    -----------
    index_constructor: Callable
        Function that constructs the index
    data: np.ndarray
        Data vectors
    params: Dict[str, Any]
        Parameters for the index constructor
        
    Returns:
    --------
    Tuple[Any, float]: (index, construction_time)
    """
    start_time = time.time()
    index = index_constructor(data, **params)
    construction_time = time.time() - start_time
    
    return index, construction_time

def benchmark_query_performance(
    index: Any,
    query_func: Callable,
    queries: np.ndarray,
    k: int,
    ground_truth: np.ndarray = None
) -> Dict[str, Any]:
    """
    Benchmark query performance
    
    Parameters:
    -----------
    index: Any
        Index to query
    query_func: Callable
        Function that queries the index
    queries: np.ndarray
        Query vectors
    k: int
        Number of neighbors to find
    ground_truth: np.ndarray
        Ground truth indices (optional)
        
    Returns:
    --------
    Dict[str, Any]: Performance metrics
    """
    n_queries = queries.shape[0]
    
    # Measure query time
    start_time = time.time()
    results = []
    
    for i in range(n_queries):
        result = query_func(index, queries[i], k)
        results.append(result)
        
    query_time = time.time() - start_time
    
    # Fix for inhomogeneous array shapes
    # Make sure all results have the same length by padding or truncating
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
    results_array = np.array(processed_results)
        
    # Calculate recall if ground truth is provided
    recall = None
    if ground_truth is not None:
        recall = evaluate_recall(ground_truth, results_array)
        
    return {
        "total_query_time": query_time,
        "avg_query_time": query_time / n_queries,
        "qps": n_queries / query_time,
        "recall": recall
    }

def plot_recall_vs_qps(results: List[Dict[str, Any]], title: str = "Recall vs. QPS") -> None:
    """
    Plot recall vs. queries per second
    
    Parameters:
    -----------
    results: List[Dict[str, Any]]
        List of benchmark results
    title: str
        Plot title
    """
    recalls = [r["recall"] for r in results]
    qps = [r["qps"] for r in results]
    labels = [r.get("label", f"Config {i}") for i, r in enumerate(results)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(recalls, qps, s=100)
    
    for i, label in enumerate(labels):
        plt.annotate(label, (recalls[i], qps[i]), fontsize=12)
        
    plt.xlabel("Recall@k", fontsize=14)
    plt.ylabel("Queries per second", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.close()

def plot_construction_time_vs_params(
    results: List[Dict[str, Any]],
    param_name: str,
    title: str = "Construction Time vs. Parameter"
) -> None:
    """
    Plot construction time vs. parameter value
    
    Parameters:
    -----------
    results: List[Dict[str, Any]]
        List of benchmark results
    param_name: str
        Parameter name to plot against
    title: str
        Plot title
    """
    param_values = [r["params"][param_name] for r in results]
    times = [r["construction_time"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, times, 'o-', linewidth=2, markersize=10)
    
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel("Construction Time (s)", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.close()

def analyze_complexity(
    index: Any,
    queries: np.ndarray,
    query_func: Callable,
    k: int,
    dataset_sizes: List[int]
) -> Dict[str, Any]:
    """
    Analyze search complexity by measuring query time vs. dataset size
    
    Parameters:
    -----------
    index: Any
        Index to query
    queries: np.ndarray
        Query vectors
    query_func: Callable
        Function that queries the index
    k: int
        Number of neighbors to find
    dataset_sizes: List[int]
        List of dataset sizes to test
        
    Returns:
    --------
    Dict[str, Any]: Complexity analysis results
    """
    n_queries = queries.shape[0]
    times = []
    
    for size in dataset_sizes:
        # Limit the index to the first 'size' elements
        index.limit_elements = size
        
        # Measure query time
        start_time = time.time()
        
        for i in range(n_queries):
            query_func(index, queries[i], k)
            
        query_time = time.time() - start_time
        times.append(query_time / n_queries)
        
    # Reset index size
    index.limit_elements = None
    
    # Plot complexity
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, times, 'o-', linewidth=2, markersize=10)
    
    plt.xlabel("Dataset Size (N)", fontsize=14)
    plt.ylabel("Average Query Time (s)", fontsize=14)
    plt.title("Query Time Complexity", fontsize=16)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    # Fit log curve to confirm O(log N) complexity
    log_sizes = np.log(dataset_sizes)
    coeffs = np.polyfit(log_sizes, times, 1)
    
    # Plot the fitted curve
    fit_times = coeffs[0] * log_sizes + coeffs[1]
    plt.plot(dataset_sizes, np.exp(fit_times), 'r--', linewidth=2, 
             label=f'Fit: {coeffs[0]:.4f} * log(N) + {coeffs[1]:.4f}')
    
    plt.legend(fontsize=12)
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/query_time_complexity.png")
    plt.close()
    
    return {
        "dataset_sizes": dataset_sizes,
        "query_times": times,
        "fit_coefficients": coeffs
    } 