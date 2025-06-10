import weaviate
from weaviate.embedded import EmbeddedOptions
import numpy as np

def main():
    print("Testing Weaviate client v4...")
    
    # Setup embedded Weaviate
    print("Setting up embedded Weaviate client...")
    embedded_options = EmbeddedOptions(
        persistence_data_path="./weaviate-test-data",
        additional_env_vars={"ENABLE_MODULES": "text2vec-transformers"}
    )
    
    client = weaviate.WeaviateClient(
        embedded_options=embedded_options
    )
    
    # Check if client is ready
    print("Checking if Weaviate is ready...")
    try:
        ready = client.is_ready()
        print(f"Weaviate is ready: {ready}")
    except Exception as e:
        print(f"Error checking if Weaviate is ready: {e}")
        return
    
    # Create a test collection
    collection_name = "TestCollection"
    print(f"Creating collection: {collection_name}")
    
    # Check if collection exists and delete if it does
    try:
        client.collections.delete(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        print(f"Collection {collection_name} does not exist yet")
    
    # Create collection
    try:
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=weaviate.classes.Configure.Vectorizer.none(),
            properties=[
                weaviate.classes.Property(
                    name="test_prop",
                    data_type=weaviate.classes.DataType.TEXT
                )
            ],
            vector_index_config=weaviate.classes.Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.L2_SQUARED
            )
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        print(f"Error creating collection: {e}")
        return
    
    # Add a test object
    print("Adding test object...")
    try:
        # Generate a random vector
        test_vector = np.random.rand(128).astype(np.float32).tolist()
        
        # Add object
        result = collection.data.insert(
            properties={"test_prop": "test value"},
            vector=test_vector
        )
        print(f"Added object with ID: {result}")
    except Exception as e:
        print(f"Error adding object: {e}")
        return
    
    # Query the object
    print("Querying objects...")
    try:
        objects = collection.query.get(
            ["test_prop"]
        ).do()
        print(f"Query result: {objects}")
    except Exception as e:
        print(f"Error querying objects: {e}")
    
    print("Weaviate client v4 test completed successfully!")

if __name__ == "__main__":
    main() 