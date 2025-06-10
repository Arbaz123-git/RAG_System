# Vector Database Benchmarking

This document outlines how to benchmark our pure-Python HNSW implementation against Weaviate, a production-ready vector database.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Weaviate using Docker Compose:
```bash
docker-compose up -d
```

## Running the Benchmarks

### Generate Embeddings (if not already available)

For text embeddings:
```bash
python embeddings/main.py --csv_path path/to/your/text/data.csv --output_path embeddings/text_embeddings.npy
```

For image embeddings:
```bash
python embeddings/image_embeddings.py --image_dir path/to/your/images --output_path embeddings/image_embeddings.npy
```

### Run the Benchmark

To benchmark both text and image embeddings:
```bash
python vector_db_benchmark.py --text_embeddings embeddings/text_embeddings.npy --image_embeddings embeddings/image_embeddings.npy
```

To benchmark only text embeddings:
```bash
python vector_db_benchmark.py --mode text --text_embeddings embeddings/text_embeddings.npy
```

To benchmark only image embeddings:
```bash
python vector_db_benchmark.py --mode image --image_embeddings embeddings/image_embeddings.npy
```

### Additional Options

- `--n_queries`: Number of queries to use for benchmarking (default: 100)
- `--k`: Number of nearest neighbors to retrieve (default: 10)
- `--hnsw_params`: HNSW parameters as JSON string (default: '{"M": 16, "ef_construction": 200, "ef_search": 100}')
- `--output_dir`: Directory to save results (default: "benchmark_results")

Example with custom parameters:
```bash
python vector_db_benchmark.py --mode both --n_queries 200 --k 10 --hnsw_params '{"M": 32, "ef_construction": 300, "ef_search": 150}'
```

## Benchmark Results

The benchmarking script will generate the following outputs:

1. CSV file with results: `benchmark_results.csv`
2. Markdown table with results: `benchmark_results.md`
3. Visualization of results: `benchmark_results.png`
4. Full detailed results in JSON format: `benchmark_results/full_results.json`

The main metrics measured are:
- Average query time (ms)
- P95 latency (ms)
- Recall@k (compared to exact nearest neighbors)
- Index build time (s)
- Index save/load time (s) (for HNSW only)

## Expected Table Format

| Backend              | Avg. Query Time (ms) | p95 Latency (ms) | Recall@10 |
| -------------------- | -------------------- | ---------------- | ---------- |
| Pure-Python HNSW     | …                    | …                | ≥ 0.92     |
| Weaviate             | …                    | …                | ≥ 0.92     | 