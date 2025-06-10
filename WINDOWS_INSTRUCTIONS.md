# Running Vector Database Benchmarks on Windows

Since Weaviate's embedded mode is not supported on Windows, you'll need to run Weaviate in Docker. This document provides step-by-step instructions.

## Prerequisites

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Weaviate in Docker

1. Start Docker Desktop
2. Open a terminal and run:
   ```bash
   docker-compose up -d
   ```
   This will start Weaviate using the configuration in the `docker-compose.yml` file.

3. Verify that Weaviate is running:
   ```bash
   docker ps
   ```
   You should see a container named `multimodal_rag_microservices-weaviate-1` (or similar) running.

## Running the Benchmarks

The benchmark script has been updated to automatically detect Windows and use the Docker-based Weaviate instance instead of embedded mode.

1. Run the benchmark script:
   ```bash
   python vector_db_benchmark.py --text_embeddings embeddings/text_embeddings.npy --image_embeddings embeddings/image_embeddings.npy
   ```

2. For a smaller test run:
   ```bash
   python vector_db_benchmark.py --mode text --text_embeddings embeddings/text_embeddings.npy --n_queries 10
   ```

## Troubleshooting

If you encounter connection errors to Weaviate:

1. Make sure Docker is running
2. Check if the Weaviate container is running:
   ```bash
   docker ps
   ```
3. If not running, check the logs:
   ```bash
   docker logs multimodal_rag_microservices-weaviate-1
   ```
4. Try restarting the container:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## Generating Embeddings

If you need to generate embeddings first:

For text embeddings:
```bash
python embeddings/main.py --csv_path path/to/your/text/data.csv --output_path embeddings/text_embeddings.npy
```

For image embeddings:
```bash
python embeddings/image_embeddings_fixed.py --image_dir data/kvasir-seg/Kvasir-SEG/images --mask_dir data/kvasir-seg/Kvasir-SEG/masks --output_path embeddings/image_embeddings.npy --limit 100 --use_hf
```

## Expected Output

The benchmark will generate several files:
- `benchmark_results.csv`: CSV file with performance metrics
- `benchmark_results.md`: Markdown table with performance metrics
- `engineering_dossier_table.md`: Formatted table for the engineering dossier
- `benchmark_results.png`: Visualization of performance metrics
- `benchmark_results/full_results.json`: Detailed results in JSON format 