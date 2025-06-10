# Multimodal RAG Agent for Medical Image Analysis

This project implements a multimodal retrieval-augmented generation (RAG) pipeline for medical image analysis, specifically focused on colonoscopy images and related medical text.

## Overview

The implementation consists of two main agents:

1. **Retriever Agent**: Embeds user queries using a sentence transformer model, then retrieves relevant text sentences and image patches from a vector database (Weaviate).

2. **Reasoner Agent**: Combines the retrieved text and image data to generate a coherent response that:
   - Directly quotes relevant text sources
   - Explains the significance of selected images
   - Provides citations for all sources used

## Features

- Text and image retrieval from a vector database
- Semantic search using sentence embeddings
- Multimodal prompt engineering (text + image)
- Mock response generation for demonstration
- LLM-based reasoning with GROQ API (when available)

## Files

- `multimodal_rag_direct.py`: Simplified implementation that works without LangGraph
- `multimodal_rag_agent.py`: LangGraph implementation (may require additional debugging)
- `requirements_rag.txt`: Required dependencies

## Prerequisites

- Python 3.8+
- Docker (for running Weaviate)
- Pre-embedded text and images in Weaviate
- GROQ API key (optional, for LLM responses)

## Installation

1. Install dependencies:
   ```
   pip install -r requirements_rag.txt
   ```

2. Run Weaviate (if not using mock data):
   ```
   docker run -d -p 8080:8080 -p 50051:50051 --name weaviate -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH="./data" -e CLUSTER_HOSTNAME="node1" semitechnologies/weaviate:1.24.1
   ```

3. Set GROQ API key (optional):
   ```
   set GROQ_API_KEY=your-api-key-here
   ```

## Usage

Run the direct implementation:
```
python multimodal_rag_direct.py
```

The script will process example queries and show:
- Retrieved text snippets with relevance scores
- Retrieved images with relevance scores
- The most relevant selected image
- A generated response that combines this information

## Example Queries

The implementation includes these example queries:
1. "What are the characteristics of polyps in colonoscopy images?"
2. "How can imaging help in identifying early-stage colon cancer?"
3. "What's the difference between benign and malignant polyps?"

## Notes

- Without a GROQ API key, the system will use pre-defined mock responses
- The LangGraph implementation may require additional debugging
- Vector dimension mismatches may occur if the embedded dimensions don't match

## Future Improvements

- Add support for more imaging modalities
- Implement better error handling for Weaviate connection issues
- Enhance the LangGraph implementation with branching logic
- Add support for multiple LLM providers beyond GROQ 