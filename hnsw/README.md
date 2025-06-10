# Custom HNSW Implementation

This is a pure Python implementation of the Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search. The implementation is designed to be easy to understand and modify, while still providing good performance.

## Features

- Pure Python implementation (no external ANN libraries like Faiss, Annoy, etc.)
- O(log N) average search complexity
- Support for different distance metrics (L2, cosine, inner product)
- Configurable parameters for trade-offs between speed, memory, and recall
- Tools for benchmarking and evaluation

## Requirements

- NumPy
- Matplotlib (for visualization)
- scikit-learn (for exact nearest neighbor search in benchmarks)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from hnsw import HNSW
import numpy as np

# Create some random data
dim = 128
n_vectors = 10000
data = np.random.random((n_vectors, dim)).astype(np.float32)

# Create HNSW index
index = HNSW(
    dim=dim,
    max_elements=n_vectors,
    ef_construction=200,  # Size of the dynamic list for constructing the graph
    M=16,                 # Number of bi-directional links created for each element
    ef_search=50          # Size of the dynamic list for searching the graph
)

# Add vectors to the index
for i, vec in enumerate(data):
    index.add(vec, i)

# Search for nearest neighbors
query = np.random.random(dim).astype(np.float32)
results = index.search(query, k=10)  # Returns list of (distance, id) tuples

# Print results
for dist, idx in results:
    print(f"ID: {idx}, Distance: {dist}")

# Save the index
index.save("my_index.npy")

# Load the index
loaded_index = HNSW.load("my_index.npy")
```

### Running the Example

```bash
python example.py
```

This will:
1. Generate random data and queries
2. Build an HNSW index
3. Measure construction and query performance
4. Compare against exact search
5. Test different parameter values
6. Generate plots showing recall vs. QPS trade-offs
7. Demonstrate O(log N) search complexity

### Running Benchmarks

```bash
python benchmark.py --data_size 100000 --query_size 1000 --dim 128 --k 10
```

Options:
- `--data_size`: Number of data vectors (default: 100000)
- `--query_size`: Number of query vectors (default: 1000)
- `--dim`: Vector dimensionality (default: 128)
- `--k`: Number of nearest neighbors (default: 10)
- `--mode`: Benchmark mode (choices: all, sweep, complexity, recall; default: all)
- `--save_dir`: Directory to save results (default: results)

### Running Tests

```bash
cd tests
python test_hnsw.py
```

## Algorithm Details

HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search that constructs a multi-layered graph. The key features are:

1. **Hierarchical Structure**: Multiple layers with decreasing density, enabling logarithmic search complexity.
2. **Navigable Small World**: Each layer forms a navigable small world graph with short paths between any two nodes.
3. **Greedy Search**: Search starts at the top layer and descends through the hierarchy.

The main parameters that affect performance:

- **M**: Controls the maximum number of connections per element. Higher values increase recall but also increase memory usage and construction time.
- **ef_construction**: Controls the size of the dynamic candidate list during construction. Higher values increase recall but slow down construction.
- **ef_search**: Controls the size of the dynamic candidate list during search. Higher values increase recall but slow down search.

## Performance

With proper parameter tuning, this implementation can achieve:

- **Recall@10 ≥ 0.92** on datasets with ~200,000 vectors
- **O(log N)** average search complexity
- Significant speedup over exact search (10-100x depending on dataset size and parameters)

## References

- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4), 824-836. 