# Embedding Pipeline for Text and Images

This directory contains scripts for generating embeddings from text and images, specifically designed for:
1. Sentence-level embeddings from PubMed abstracts
2. Patch-level embeddings from Kvasir-SEG polyp images

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data

The pipeline expects the following data:

### Text Data
- PubMed abstracts in CSV format at `data/train.csv/train.csv`
- We process the first 200,000 records from the `abstract_text` column

### Image Data
- Kvasir-SEG polyp images at `data/kvasir-seg/Kvasir-SEG/images/`
- Corresponding masks at `data/kvasir-seg/Kvasir-SEG/masks/`
- We process the first 1,000 images

## Usage

### Generate Both Text and Image Embeddings

To generate both text and image embeddings with default settings:

```bash
python embeddings/main.py --output_dir output
```

### Generate Only Text Embeddings

```bash
python embeddings/text_embeddings.py --csv_path data/train.csv/train.csv --output_path output/sentence_embeddings.npy
```

### Generate Only Image Embeddings

```bash
python embeddings/image_embeddings.py --image_dir data/kvasir-seg/Kvasir-SEG/images --mask_dir data/kvasir-seg/Kvasir-SEG/masks --output_path output/image_embeddings.npy
```

## Configuration Options

### Text Embeddings

- `--csv_path`: Path to the CSV file containing abstracts (default: data/train.csv/train.csv)
- `--output_path`: Path to save the embeddings (default: sentence_embeddings.npy)
- `--model_name`: SentenceTransformers model to use (default: all-mpnet-base-v2)
- `--batch_size`: Batch size for embedding (default: 32)
- `--num_records`: Number of records to process (default: 200000)
- `--device`: Device to use (cpu, cuda:0, etc.)

### Image Embeddings

- `--image_dir`: Directory containing images (required)
- `--mask_dir`: Directory containing masks (optional)
- `--output_path`: Path to save the embeddings (default: image_embeddings.npy)
- `--model_name`: SentenceTransformers model to use (default: clip-ViT-B-32)
- `--batch_size`: Batch size for embedding (default: 32)
- `--patch_size`: Size of patches (default: 224, use 0 for whole images)
- `--limit`: Limit the number of images to process (default: 1000)
- `--device`: Device to use (cpu, cuda:0, etc.)

## Output Files

The pipeline generates the following files:

### Text Embeddings
- `sentence_embeddings.npy`: NumPy array of shape [#sentences × embedding_dim]
- `sentence_embeddings_mapping.npy`: Mapping from sentences to original abstracts
- `sentence_embeddings_sentences.txt`: Original sentences for reference

### Image Embeddings
- `image_embeddings.npy`: NumPy array of shape [#patches × embedding_dim]
- `image_embeddings_metadata.npy`: Metadata for each patch (image path, coordinates)

## Model Details

### Text Embedding Model
- Model: all-mpnet-base-v2
- Embedding dimension: 768
- Description: State-of-the-art sentence embedding model from SentenceTransformers

### Image Embedding Model
- Model: clip-ViT-B-32
- Embedding dimension: 512
- Description: Vision Transformer model from OpenAI CLIP, pretrained on a large dataset of image-text pairs 