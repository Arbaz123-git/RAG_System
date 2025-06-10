# Technical Details: Text and Image Embedding Pipeline

This document outlines the technical decisions and implementation details of our embedding pipeline for text and images.

## Text Embedding Implementation

### Model Selection
We chose the `all-mpnet-base-v2` model from SentenceTransformers for text embedding because:

1. **Performance**: It achieves state-of-the-art performance on semantic textual similarity tasks
2. **Dimension**: It produces 768-dimensional embeddings, which provide a good balance between expressiveness and computational efficiency
3. **Medical Text Compatibility**: While not specifically fine-tuned for medical text, this model has been shown to perform well on various domains including scientific and medical text

### Text Processing Pipeline
1. **Loading Data**: We load the PubMed abstracts from the CSV file, limiting to the first 200,000 records
2. **Sentence Splitting**: We use NLTK's `sent_tokenize` to split each abstract into individual sentences
3. **Embedding Generation**: Each sentence is embedded independently using the SentenceTransformer model
4. **Tracking**: We maintain a mapping between sentences and their original abstracts for potential downstream tasks

### Optimizations
- **Batch Processing**: Sentences are processed in batches to maximize GPU utilization
- **Progress Tracking**: A progress bar is displayed during embedding generation
- **Memory Efficiency**: We process abstracts sequentially to avoid loading all text into memory at once

## Image Embedding Implementation

### Model Selection
We chose the `clip-ViT-B-32` model from SentenceTransformers for image embedding because:

1. **Vision Transformer Architecture**: ViT models have shown excellent performance on vision tasks
2. **CLIP Training**: The model was trained on a diverse dataset of image-text pairs, making it robust for various domains
3. **Dimension**: It produces 512-dimensional embeddings, which is sufficient for capturing visual features
4. **Transfer Learning**: The pre-training on a large dataset makes it suitable for medical images without specific fine-tuning

### Image Processing Pipeline
1. **Loading Images**: We load images from the specified directory, limiting to the first 1,000 images
2. **Patch Extraction**: Images are divided into fixed-size patches (224×224 pixels) to capture local features
3. **Mask Integration**: Corresponding segmentation masks are loaded if available
4. **Embedding Generation**: Each patch is embedded independently using the SentenceTransformer model
5. **Metadata Tracking**: We maintain metadata about each patch's source image and position

### Optimizations
- **Lazy Loading**: Images are loaded only when needed during batch processing
- **Patch Precomputation**: Patch information is precomputed to determine the total dataset size in advance
- **DataLoader**: PyTorch's DataLoader is used with multiple workers for parallel processing
- **GPU Acceleration**: Batch processing on GPU for faster embedding generation

## Performance Considerations

### Memory Usage
- **Text Embeddings**: For 200,000 abstracts with an average of 5 sentences each, the resulting embedding matrix would be approximately:
  - 1,000,000 sentences × 768 dimensions × 4 bytes ≈ 3.1 GB
- **Image Embeddings**: For 1,000 images with an average of 9 patches each (3×3 grid), the resulting embedding matrix would be approximately:
  - 9,000 patches × 512 dimensions × 4 bytes ≈ 18.4 MB

### Computational Requirements
- **Text Processing**: Embedding generation takes approximately 1-2 hours on a modern GPU
- **Image Processing**: Embedding generation takes approximately 10-15 minutes on a modern GPU

## Extensibility

The pipeline is designed to be easily extensible:

1. **Alternative Models**: Different SentenceTransformer models can be specified via command-line arguments
2. **Patch Size Customization**: Image patch size can be adjusted based on the specific requirements
3. **Whole Image Processing**: Setting `patch_size=0` processes whole images instead of patches
4. **Device Selection**: The pipeline can run on CPU or GPU based on availability and user preference

## Future Improvements

Potential enhancements to consider:

1. **Domain-Specific Fine-tuning**: Fine-tune the text embedding model on medical literature for improved performance
2. **Multi-GPU Support**: Add support for distributed processing across multiple GPUs
3. **Incremental Processing**: Enable resuming from a checkpoint if processing is interrupted
4. **Dimensionality Reduction**: Add options for PCA or other dimensionality reduction techniques
5. **Quality Assessment**: Implement evaluation metrics to assess embedding quality 