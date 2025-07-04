#!/usr/bin/env python
"""
Simple Weaviate Schema Creation Script using direct REST API
This script creates a schema in Weaviate for storing text and image embeddings.
"""

import requests
import json
import time

# Constants
WEAVIATE_URL = "http://localhost:8080"
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"

def check_weaviate_connection():
    """Check if Weaviate is running and accessible."""
    print("Checking Weaviate connection...")
    
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/meta")
        if response.status_code == 200:
            print("✅ Successfully connected to Weaviate")
            return True
        else:
            print(f"❌ Failed to connect to Weaviate: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        print("Make sure Weaviate is running (docker ps)")
        return False

def get_schema():
    """Get the current Weaviate schema."""
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/schema")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get schema: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Failed to get schema: {e}")
        return None

def delete_class(class_name):
    """Delete a class from the schema."""
    try:
        response = requests.delete(f"{WEAVIATE_URL}/v1/schema/{class_name}")
        if response.status_code == 200:
            print(f"✅ Successfully deleted class {class_name}")
            return True
        else:
            print(f"❌ Failed to delete class {class_name}: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to delete class {class_name}: {e}")
        return False

def create_text_collection():
    """Create the text embeddings collection."""
    print("\nCreating Text Embeddings Collection...")
    
    # Schema for text collection
    schema = {
        "class": TEXT_COLLECTION_NAME,
        "description": "Collection for text embeddings",
        "vectorizer": "none",  # Use custom vectors
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "properties": [
            {
                "name": "embedding_id",
                "dataType": ["int"],
                "description": "ID of the embedding in the original dataset"
            },
            {
                "name": "text",
                "dataType": ["text"],
                "description": "Original text content"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/schema",
            json=schema
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully created {TEXT_COLLECTION_NAME} collection")
            return True
        else:
            print(f"❌ Failed to create {TEXT_COLLECTION_NAME} collection: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Failed to create {TEXT_COLLECTION_NAME} collection: {e}")
        return False

def create_image_collection():
    """Create the image embeddings collection."""
    print("\nCreating Image Embeddings Collection...")
    
    # Schema for image collection
    schema = {
        "class": IMAGE_COLLECTION_NAME,
        "description": "Collection for image embeddings",
        "vectorizer": "none",  # Use custom vectors
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "properties": [
            {
                "name": "embedding_id",
                "dataType": ["int"],
                "description": "ID of the embedding in the original dataset"
            },
            {
                "name": "image_path",
                "dataType": ["text"],
                "description": "Path to the original image"
            },
            {
                "name": "region",
                "dataType": ["text"],
                "description": "Region of interest in the image"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/schema",
            json=schema
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully created {IMAGE_COLLECTION_NAME} collection")
            return True
        else:
            print(f"❌ Failed to create {IMAGE_COLLECTION_NAME} collection: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Failed to create {IMAGE_COLLECTION_NAME} collection: {e}")
        return False

def main():
    """Main function to create Weaviate schema."""
    print("=" * 50)
    print("Creating Weaviate Schema (Simple Version)")
    print("=" * 50)
    
    # Check Weaviate connection
    if not check_weaviate_connection():
        print("❌ Exiting due to connection error")
        return
    
    # Get current schema
    schema = get_schema()
    if schema:
        print("Current schema retrieved")
        
        # Check if collections exist
        classes = [c["class"] for c in schema.get("classes", [])]
        
        if TEXT_COLLECTION_NAME in classes:
            print(f"Collection {TEXT_COLLECTION_NAME} already exists.")
            delete = input(f"Do you want to delete {TEXT_COLLECTION_NAME} and recreate it? (y/n): ").strip().lower()
            if delete == 'y':
                delete_class(TEXT_COLLECTION_NAME)
                time.sleep(1)  # Give Weaviate time to process
        
        if IMAGE_COLLECTION_NAME in classes:
            print(f"Collection {IMAGE_COLLECTION_NAME} already exists.")
            delete = input(f"Do you want to delete {IMAGE_COLLECTION_NAME} and recreate it? (y/n): ").strip().lower()
            if delete == 'y':
                delete_class(IMAGE_COLLECTION_NAME)
                time.sleep(1)  # Give Weaviate time to process
    
    # Create collections
    text_created = create_text_collection()
    image_created = create_image_collection()
    
    if text_created and image_created:
        print("\n✅ Schema created successfully")
        print("\nNext steps:")
        print("1. Run the upload_embeddings.py script to populate Weaviate with your data")
        print("2. Use the multimodal_rag_with_groq.py script to query your data")
    else:
        print("\n❌ Schema creation failed")
        print("Please check the error messages and try again")

if __name__ == "__main__":
    main()
