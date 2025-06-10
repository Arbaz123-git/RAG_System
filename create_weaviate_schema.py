#!/usr/bin/env python
"""
Create Weaviate Schema Script for MultiModal RAG
This script creates the necessary schema in Weaviate for storing text and image embeddings.
"""

import os
import weaviate
import time
from sentence_transformers import SentenceTransformer

# Constants
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model for embeddings
EMBEDDING_DIMENSION = 384  # Embedding dimension for all-MiniLM-L6-v2

def setup_weaviate_client():
    """Set up and connect to Weaviate client."""
    print("Connecting to Weaviate...")
    client = weaviate.WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_url(
            "http://localhost:8080",
            grpc_port=50051
        )
    )
    
    # Check connection
    print("Testing connection to Weaviate...")
    try:
        client.connect()
        print("✅ Successfully connected to Weaviate")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        print("Make sure Weaviate is running (docker ps)")
        return None

def check_and_delete_existing_collections(client):
    """Check if collections already exist and delete them if needed."""
    print("\nChecking for existing collections...")
    
    try:
        # Use REST API to get schema
        collections = []
        schema = client.schema.get()
        if 'classes' in schema:
            collections = [c.get('class') for c in schema.get('classes', [])]
        
        if TEXT_COLLECTION_NAME in collections:
            print(f"Collection {TEXT_COLLECTION_NAME} already exists.")
            delete = input(f"Do you want to delete {TEXT_COLLECTION_NAME} and recreate it? (y/n): ").strip().lower()
            if delete == 'y':
                print(f"Deleting {TEXT_COLLECTION_NAME}...")
                client.schema.delete_class(TEXT_COLLECTION_NAME)
                print(f"✅ Deleted {TEXT_COLLECTION_NAME}")
                time.sleep(1)  # Give Weaviate time to process
        
        if IMAGE_COLLECTION_NAME in collections:
            print(f"Collection {IMAGE_COLLECTION_NAME} already exists.")
            delete = input(f"Do you want to delete {IMAGE_COLLECTION_NAME} and recreate it? (y/n): ").strip().lower()
            if delete == 'y':
                print(f"Deleting {IMAGE_COLLECTION_NAME}...")
                client.schema.delete_class(IMAGE_COLLECTION_NAME)
                print(f"✅ Deleted {IMAGE_COLLECTION_NAME}")
                time.sleep(1)  # Give Weaviate time to process
    
    except Exception as e:
        print(f"❌ Error checking collections: {e}")

def create_text_collection(client):
    """Create the text embeddings collection in Weaviate."""
    print("\nCreating Text Embeddings Collection...")
    
    try:
        # Create schema class definition for Weaviate 4.15.0
        class_obj = {
            "class": TEXT_COLLECTION_NAME,
            "description": "Collection for text embeddings",
            "vectorizer": "none",  # Use "none" vectorizer for custom vectors
            "vectorIndexType": "hnsw",  # Use HNSW index
            "vectorIndexConfig": {
                "distance": "cosine"  # Use cosine distance
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
        
        # Create the class
        client.schema.create_class(class_obj)
        print(f"✅ Created {TEXT_COLLECTION_NAME} collection")
        return True
    except Exception as e:
        print(f"❌ Failed to create {TEXT_COLLECTION_NAME} collection: {e}")
        return False

def create_image_collection(client):
    """Create the image embeddings collection in Weaviate."""
    print("\nCreating Image Embeddings Collection...")
    
    try:
        # Create schema class definition for Weaviate 4.15.0
        class_obj = {
            "class": IMAGE_COLLECTION_NAME,
            "description": "Collection for image embeddings",
            "vectorizer": "none",  # Use "none" vectorizer for custom vectors
            "vectorIndexType": "hnsw",  # Use HNSW index
            "vectorIndexConfig": {
                "distance": "cosine"  # Use cosine distance
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
        
        # Create the class
        client.schema.create_class(class_obj)
        print(f"✅ Created {IMAGE_COLLECTION_NAME} collection")
        return True
    except Exception as e:
        print(f"❌ Failed to create {IMAGE_COLLECTION_NAME} collection: {e}")
        return False

def test_schema_with_sample_data(client):
    """Test the schema by adding a sample object to each collection."""
    print("\nTesting schema with sample data...")
    
    # Create a sentence transformer model to get embedding dimension
    print("Loading sentence transformer model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    sample_text = "This is a test sentence for embedding."
    sample_embedding = model.encode(sample_text).tolist()
    
    try:
        # Add sample text embedding
        # Use Weaviate 4.15.0 API to add data
        data_object = {
            "embedding_id": 0,
            "text": sample_text
        }
        
        # Add data to text collection
        text_result = client.data_object.create(
            data_object,
            TEXT_COLLECTION_NAME,
            vector=sample_embedding
        )
        print(f"✅ Successfully added sample data to {TEXT_COLLECTION_NAME}")
        
        # Add sample image embedding
        image_object = {
            "embedding_id": 0,
            "image_path": "sample/image.jpg",
            "region": "full"
        }
        
        # Add data to image collection
        image_result = client.data_object.create(
            image_object,
            IMAGE_COLLECTION_NAME,
            vector=sample_embedding
        )
        print(f"✅ Successfully added sample data to {IMAGE_COLLECTION_NAME}")
        
        # Test retrieval using search
        print("Testing vector search...")
        query_result = client.query.get(
            TEXT_COLLECTION_NAME, 
            ["embedding_id", "text"]
        ).with_near_vector({
            "vector": sample_embedding
        }).with_limit(1).do()
        
        # Check if we got a result
        if query_result and 'data' in query_result and 'Get' in query_result['data']:
            objects = query_result['data']['Get'].get(TEXT_COLLECTION_NAME, [])
            if objects:
                print("✅ Successfully retrieved data using vector search")
                return True
        
        print("❌ Failed to retrieve data using vector search")
        return False
    
    except Exception as e:
        print(f"❌ Error testing schema: {e}")
        return False

def main():
    """Main function to create Weaviate schema."""
    print("=" * 50)
    print("Creating Weaviate Schema for MultiModal RAG")
    print("=" * 50)
    
    # Connect to Weaviate
    client = setup_weaviate_client()
    if not client:
        print("❌ Exiting due to connection error")
        return
    
    # Check and delete existing collections if needed
    check_and_delete_existing_collections(client)
    
    # Create collections
    text_created = create_text_collection(client)
    image_created = create_image_collection(client)
    
    if text_created and image_created:
        # Test schema with sample data
        test_success = test_schema_with_sample_data(client)
        
        if test_success:
            print("\n✅ Schema creation and testing completed successfully")
            print("You're now ready to upload your embeddings to Weaviate")
            print("\nNext steps:")
            print("1. Run the upload_embeddings.py script to populate Weaviate with your data")
            print("2. Use the multimodal_rag_with_groq.py script to query your data")
        else:
            print("\n❌ Schema testing failed")
            print("Please check the error messages and try again")
    else:
        print("\n❌ Schema creation failed")
        print("Please check the error messages and try again")
    
    # Close connection
    client.close()

if __name__ == "__main__":
    main() 