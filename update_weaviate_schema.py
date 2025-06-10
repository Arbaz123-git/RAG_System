#!/usr/bin/env python
"""
Update Weaviate Schema Script
This script updates the existing schema in Weaviate with additional properties.
"""

import requests
import json

# Constants
WEAVIATE_URL = "http://localhost:8080"
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"

def print_header(message):
    """Print a header message with decoration."""
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50)

def check_weaviate_connection():
    """Check if Weaviate is running and accessible."""
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

def add_property_to_class(class_name, property_name, data_type, description):
    """Add a property to an existing class."""
    print(f"\nAdding property '{property_name}' to class '{class_name}'...")
    
    # Property schema
    property_schema = {
        "dataType": data_type,
        "name": property_name,
        "description": description,
        "indexFilterable": True,
        "indexSearchable": True
    }
    
    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/schema/{class_name}/properties",
            json=property_schema
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully added property '{property_name}' to class '{class_name}'")
            return True
        else:
            print(f"❌ Failed to add property: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Failed to add property: {e}")
        return False

def main():
    """Main function to update Weaviate schema."""
    print_header("Updating Weaviate Schema")
    
    # Check Weaviate connection
    if not check_weaviate_connection():
        print("❌ Exiting due to connection error")
        return
    
    # Get current schema
    schema = get_schema()
    if not schema:
        print("❌ Failed to retrieve schema")
        return
    
    # Print current schema classes
    classes = [c["class"] for c in schema.get("classes", [])]
    print(f"Current schema classes: {classes}")
    
    # Check if the required classes exist
    if TEXT_COLLECTION_NAME not in classes:
        print(f"❌ Class '{TEXT_COLLECTION_NAME}' does not exist")
        return
    
    if IMAGE_COLLECTION_NAME not in classes:
        print(f"❌ Class '{IMAGE_COLLECTION_NAME}' does not exist")
        return
    
    # Add text property to TextEmbeddings class if it doesn't exist
    text_properties = next((c["properties"] for c in schema["classes"] if c["class"] == TEXT_COLLECTION_NAME), [])
    text_property_names = [p["name"] for p in text_properties]
    
    if "text" not in text_property_names:
        add_property_to_class(
            TEXT_COLLECTION_NAME,
            "text",
            ["text"],
            "Original text content"
        )
    else:
        print(f"✅ Property 'text' already exists in class '{TEXT_COLLECTION_NAME}'")
    
    # Add image_path and region properties to ImageEmbeddings class if they don't exist
    image_properties = next((c["properties"] for c in schema["classes"] if c["class"] == IMAGE_COLLECTION_NAME), [])
    image_property_names = [p["name"] for p in image_properties]
    
    if "image_path" not in image_property_names:
        add_property_to_class(
            IMAGE_COLLECTION_NAME,
            "image_path",
            ["text"],
            "Path to the original image"
        )
    else:
        print(f"✅ Property 'image_path' already exists in class '{IMAGE_COLLECTION_NAME}'")
    
    if "region" not in image_property_names:
        add_property_to_class(
            IMAGE_COLLECTION_NAME,
            "region",
            ["text"],
            "Region of interest in the image"
        )
    else:
        print(f"✅ Property 'region' already exists in class '{IMAGE_COLLECTION_NAME}'")
    
    print("\n✅ Schema update completed")
    print("\nNext steps:")
    print("1. Run the upload_embeddings.py script to populate Weaviate with your data")
    print("2. Use the multimodal_rag_with_groq.py script to query your data")

if __name__ == "__main__":
    main() 