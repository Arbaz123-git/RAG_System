#!/usr/bin/env python
"""
Upload Embeddings Script for MultiModal RAG
This script uploads text and image embeddings to Weaviate for use in the MultiModal RAG system.
"""

import os
import sys
import requests
import numpy as np
import json
from tqdm import tqdm
import time
from pathlib import Path

# Constants
WEAVIATE_URL = "http://localhost:8080"
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"
BATCH_SIZE = 100  # Number of embeddings to upload in a batch

def check_weaviate_connection():
    """Check if Weaviate is running and accessible using direct REST API."""
    print("Checking Weaviate connection...")
    
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/meta")
        if response.status_code == 200:
            print("✅ Successfully connected to Weaviate")
            
            # Print version information
            meta = response.json()
            version = meta.get("version", "unknown")
            print(f"Weaviate version: {version}")
            return True
        else:
            print(f"❌ Failed to connect to Weaviate: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        print("Make sure Weaviate is running (docker ps)")
        return False

def verify_collections_exist():
    """Verify that the required collections exist in Weaviate."""
    print("\nVerifying collections exist...")
    
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/schema")
        if response.status_code == 200:
            schema = response.json()
            collections = [c["class"] for c in schema.get("classes", [])]
            
            if TEXT_COLLECTION_NAME not in collections:
                print(f"❌ Collection {TEXT_COLLECTION_NAME} does not exist.")
                print("Please run create_weaviate_schema.py first.")
                return False
            
            if IMAGE_COLLECTION_NAME not in collections:
                print(f"❌ Collection {IMAGE_COLLECTION_NAME} does not exist.")
                print("Please run create_weaviate_schema.py first.")
                return False
            
            print("✅ Required collections exist.")
            return True
        else:
            print(f"❌ Failed to get schema: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error verifying collections: {e}")
        return False

def load_existing_text_embeddings():
    """Load existing text embeddings from output directory."""
    print("\nLoading existing text embeddings...")
    
    embeddings_file = "output/sentence_embeddings.npy"
    sentences_file = "output/sentence_embeddings_sentences.txt"
    mapping_file = "output/sentence_embeddings_mapping.npy"
    
    if not os.path.exists(embeddings_file) or not os.path.exists(sentences_file):
        print(f"❌ Required files not found: {embeddings_file} or {sentences_file}")
        return None, []
    
    try:
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        # Load sentences
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines()]
        
        # Load mapping if available
        mapping = None
        if os.path.exists(mapping_file):
            try:
                mapping = np.load(mapping_file, allow_pickle=True)
                print(f"✅ Loaded mapping with {len(mapping)} entries")
            except Exception as e:
                print(f"⚠️ Could not load mapping file: {e}")
        
        print(f"✅ Loaded {len(embeddings)} text embeddings and {len(sentences)} sentences")
        
        return embeddings, sentences
    except Exception as e:
        print(f"❌ Error loading text embeddings: {e}")
        return None, []

def load_existing_image_embeddings():
    """Load existing image embeddings from output directory."""
    print("\nLoading existing image embeddings...")
    
    embeddings_file = "output/image_embeddings.npy"
    metadata_file = "output/image_embeddings_metadata.npy"
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Required file not found: {embeddings_file}")
        return None, {}
    
    try:
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                metadata = np.load(metadata_file, allow_pickle=True).item()
                print(f"✅ Loaded metadata for {len(metadata)} images")
            except Exception as e:
                print(f"⚠️ Could not load metadata file: {e}")
                # Create simple metadata if not available
                metadata = {i: {"image_path": f"image_{i}.jpg", "region": "unknown"} for i in range(len(embeddings))}
        else:
            # Create simple metadata if not available
            metadata = {i: {"image_path": f"image_{i}.jpg", "region": "unknown"} for i in range(len(embeddings))}
        
        print(f"✅ Loaded {len(embeddings)} image embeddings")
        
        return embeddings, metadata
    except Exception as e:
        print(f"❌ Error loading image embeddings: {e}")
        return None, {}

def upload_text_embeddings(embeddings, sentences):
    """Upload text embeddings to Weaviate using REST API."""
    print(f"\nUploading {len(embeddings)} text embeddings to Weaviate...")
    
    if len(embeddings) != len(sentences):
        print(f"❌ Number of embeddings ({len(embeddings)}) does not match number of sentences ({len(sentences)})")
        return False
    
    # Upload in batches
    success_count = 0
    for i in tqdm(range(0, len(embeddings), BATCH_SIZE)):
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        batch_sentences = sentences[i:i+BATCH_SIZE]
        
        for j in range(len(batch_embeddings)):
            try:
                # Create object data
                object_data = {
                    "class": TEXT_COLLECTION_NAME,
                    "properties": {
                        "embedding_id": i + j,
                        "text": batch_sentences[j]
                    },
                    "vector": batch_embeddings[j].tolist()
                }
                
                # Upload to Weaviate
                response = requests.post(
                    f"{WEAVIATE_URL}/v1/objects",
                    json=object_data
                )
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"⚠️ Failed to upload text embedding {i+j}: {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"⚠️ Error uploading text embedding {i+j}: {e}")
    
    print(f"✅ Successfully uploaded {success_count} out of {len(embeddings)} text embeddings")
    return success_count > 0

def upload_image_embeddings(embeddings, metadata):
    """Upload image embeddings to Weaviate using REST API."""
    print(f"\nUploading {len(embeddings)} image embeddings to Weaviate...")
    
    # Upload in batches
    success_count = 0
    for i in tqdm(range(0, len(embeddings), BATCH_SIZE)):
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        
        for j in range(len(batch_embeddings)):
            idx = i + j
            try:
                # Get metadata for this image
                image_metadata = metadata.get(idx, {})
                image_path = image_metadata.get("image_path", f"image_{idx}.jpg")
                region = image_metadata.get("region", "unknown")
                
                # Create object data
                object_data = {
                    "class": IMAGE_COLLECTION_NAME,
                    "properties": {
                        "embedding_id": idx,
                        "image_path": image_path,
                        "region": region
                    },
                    "vector": batch_embeddings[j].tolist()
                }
                
                # Upload to Weaviate
                response = requests.post(
                    f"{WEAVIATE_URL}/v1/objects",
                    json=object_data
                )
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"⚠️ Failed to upload image embedding {idx}: {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"⚠️ Error uploading image embedding {idx}: {e}")
    
    print(f"✅ Successfully uploaded {success_count} out of {len(embeddings)} image embeddings")
    return success_count > 0

def main():
    """Main function."""
    print("=" * 50)
    print("Upload Embeddings to Weaviate")
    print("=" * 50)
    
    # Check Weaviate connection
    if not check_weaviate_connection():
        print("❌ Exiting due to connection error")
        return
    
    # Verify collections exist
    if not verify_collections_exist():
        print("❌ Exiting due to missing collections")
        return
    
    # Load text embeddings
    text_embeddings, sentences = load_existing_text_embeddings()
    if text_embeddings is None or len(text_embeddings) == 0:
        print("❌ No text embeddings to upload")
    else:
        # Upload text embeddings
        upload_text_embeddings(text_embeddings, sentences)
    
    # Load image embeddings
    image_embeddings, metadata = load_existing_image_embeddings()
    if image_embeddings is None or len(image_embeddings) == 0:
        print("❌ No image embeddings to upload")
    else:
        # Upload image embeddings
        upload_image_embeddings(image_embeddings, metadata)
    
    print("\n✅ Upload process completed")
    print("\nNext steps:")
    print("1. Run the multimodal_rag_with_groq.py script to query your data")

if __name__ == "__main__":
    main() 