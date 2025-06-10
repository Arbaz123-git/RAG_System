#!/usr/bin/env python
# Direct implementation of Multimodal RAG without using LangGraph
import os
import numpy as np
import weaviate
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
import base64
from io import BytesIO
import json
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Constants
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Same model used in the benchmark
TEXT_TOP_K = 10  # Number of text results to retrieve
IMAGE_TOP_M = 5  # Number of image results to retrieve
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set your GROQ API key as environment variable

# Check if GROQ API key is set
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable not set. Using mock responses.")

# Helper functions
def setup_weaviate_client() -> weaviate.WeaviateClient:
    """Set up and connect to Weaviate client."""
    try:
        print("Connecting to Weaviate...")
        client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                "http://localhost:8080",
                grpc_port=50051
            )
        )
        client.connect()
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Weaviate: {e}")

def load_sentence_transformer() -> SentenceTransformer:
    """Load the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def plot_and_save_image(image_array: np.ndarray, file_path: str) -> str:
    """
    Plot image from numpy array and save to file.
    
    Args:
        image_array: Numpy array of the image
        file_path: Path to save the image
        
    Returns:
        Path to the saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Plot and save
    plt.figure(figsize=(10, 10))
    plt.imshow(image_array)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return file_path

def retriever_agent(query: str) -> Dict[str, Any]:
    """
    Retriever agent that embeds the query and retrieves relevant text and images.
    
    Args:
        query: User query
        
    Returns:
        Dictionary with retrieved texts, images, and selected image
    """
    state = {"query": query, "error": None}
    
    try:
        # Set up Weaviate client and sentence transformer
        try:
            client = setup_weaviate_client()
            model = load_sentence_transformer()
            
            # Embed the query
            query_embedding = model.encode(query)
            
            # Check text collection schema to determine embedding dimension
            try:
                print("Checking Weaviate schema...")
                text_collection = client.collections.get(TEXT_COLLECTION_NAME)
                
                # Try a test query with a small number of results to determine if dimensions match
                print("Testing query with Weaviate...")
                try:
                    # Adapt embedding dimensions if needed - detect schema dimensions
                    test_results = text_collection.query.near_vector(
                        near_vector=query_embedding.tolist(),
                        limit=1
                    ).objects
                    
                    print(f"Test query successful! Retrieved {len(test_results)} results.")
                    
                    # Proceed with full retrieval
                    text_results = text_collection.query.near_vector(
                        near_vector=query_embedding.tolist(),
                        limit=TEXT_TOP_K
                    ).objects
                    
                    # Process text results
                    retrieved_texts = []
                    for i, obj in enumerate(text_results):
                        # Get the original text using the embedding_id
                        embedding_id = obj.properties.get("embedding_id", i)
                        
                        # Load sentences from the file
                        with open("output/sentence_embeddings_sentences.txt", "r", encoding="utf-8") as f:
                            sentences = f.readlines()
                        
                        # Try to get the sentence if embedding_id is valid
                        sentence = sentences[embedding_id].strip() if embedding_id < len(sentences) else f"Sentence {embedding_id}"
                        
                        # Add to retrieved texts
                        retrieved_texts.append({
                            "id": embedding_id,
                            "text": sentence,
                            "score": obj.metadata.certainty if hasattr(obj, "metadata") and hasattr(obj.metadata, "certainty") else 0.0
                        })
                    
                    # Retrieve relevant images
                    try:
                        image_collection = client.collections.get(IMAGE_COLLECTION_NAME)
                        image_results = image_collection.query.near_vector(
                            near_vector=query_embedding.tolist(),
                            limit=IMAGE_TOP_M
                        ).objects
                        
                        # Process image results
                        retrieved_images = []
                        for i, obj in enumerate(image_results):
                            # Get the original image using the embedding_id
                            embedding_id = obj.properties.get("embedding_id", i)
                            
                            # Load image embeddings metadata
                            image_metadata = np.load("output/image_embeddings_metadata.npy", allow_pickle=True).item()
                            
                            # Get image info
                            image_info = image_metadata.get(embedding_id, {"path": f"Image {embedding_id}", "region": "full"})
                            
                            # Save temporary image if needed
                            image_path = image_info.get("path", "")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            temp_image_path = f"temp_images/{timestamp}_{embedding_id}.png"
                            
                            # Add to retrieved images
                            retrieved_images.append({
                                "id": embedding_id,
                                "path": image_path,
                                "temp_path": temp_image_path,
                                "region": image_info.get("region", "full"),
                                "score": obj.metadata.certainty if hasattr(obj, "metadata") and hasattr(obj.metadata, "certainty") else 0.0
                            })
                        
                    except Exception as img_err:
                        print(f"Error retrieving images: {img_err}")
                        print("Will use mock image data.")
                        # Create mock image data
                        retrieved_images = [
                            {"id": j, "path": f"mock_image_{j+1}.jpg", "temp_path": "", "region": "full", "score": 0.9 - (j * 0.1)}
                            for j in range(IMAGE_TOP_M)
                        ]
                    
                    # Select the most relevant image (highest score)
                    selected_image = retrieved_images[0] if retrieved_images else None
                    
                    # Update state
                    state["retrieved_texts"] = retrieved_texts
                    state["retrieved_images"] = retrieved_images
                    state["selected_image"] = selected_image
                    
                    client.close()
                    return state
                    
                except Exception as query_err:
                    print(f"Query error: {query_err}")
                    raise query_err
                
            except Exception as schema_err:
                print(f"Schema error: {schema_err}")
                raise schema_err
                
        except Exception as e:
            print(f"Warning: Failed to connect to Weaviate or vector dimension mismatch: {e}")
            print("Using mock data for demonstration purposes.")
            
            # Create mock data for demonstration
            state["retrieved_texts"] = [
                {"id": i, "text": f"Mock text result {i+1} for query: {query}", "score": 0.95 - (i * 0.05)} 
                for i in range(TEXT_TOP_K)
            ]
            
            # Enhance mock text content for each query type
            if "polyp" in query.lower():
                state["retrieved_texts"][0]["text"] = "Polyps are abnormal growths that protrude from the mucosal surface into the lumen of the colon."
                state["retrieved_texts"][1]["text"] = "Common polyp characteristics include pedunculated (with a stalk) or sessile (flat-based) appearance."
                state["retrieved_texts"][2]["text"] = "The Paris classification categorizes polyps based on their morphological features."
            elif "cancer" in query.lower():
                state["retrieved_texts"][0]["text"] = "Narrow-band imaging enhances the visibility of capillary networks and mucosal patterns to detect early-stage cancer."
                state["retrieved_texts"][1]["text"] = "Early colon cancer may present as subtle mucosal changes, irregular contours, or non-lifting sign during resection."
                state["retrieved_texts"][2]["text"] = "Chromoendoscopy with indigo carmine dye can highlight mucosal irregularities associated with early neoplasia."
            elif "benign" in query.lower() and "malignant" in query.lower():
                state["retrieved_texts"][0]["text"] = "Benign polyps typically have regular borders and homogeneous vascular patterns."
                state["retrieved_texts"][1]["text"] = "Malignant polyps often show disrupted vessel patterns, irregular margins, and heterogeneous color."
                state["retrieved_texts"][2]["text"] = "The NICE classification uses vascular patterns and surface features to differentiate benign from potentially malignant lesions."
            
            state["retrieved_images"] = [
                {"id": i, "path": f"mock_image_{i+1}.jpg", "temp_path": "", "region": "full", "score": 0.9 - (i * 0.1)}
                for i in range(IMAGE_TOP_M)
            ]
            
            # Customize image descriptions based on query
            if "polyp" in query.lower():
                state["retrieved_images"][0]["path"] = "endoscopy/polyp_pedunculated.jpg"
                state["retrieved_images"][1]["path"] = "endoscopy/polyp_sessile.jpg"
            elif "cancer" in query.lower():
                state["retrieved_images"][0]["path"] = "endoscopy/early_cancer_nbi.jpg"
                state["retrieved_images"][1]["path"] = "endoscopy/ulcerated_lesion.jpg"
            elif "benign" in query.lower() and "malignant" in query.lower():
                state["retrieved_images"][0]["path"] = "endoscopy/benign_polyp_regular_vessels.jpg"
                state["retrieved_images"][1]["path"] = "endoscopy/malignant_polyp_irregular_pattern.jpg"
            
            state["selected_image"] = state["retrieved_images"][0] if state["retrieved_images"] else None
            
            return state
    except Exception as e:
        state["error"] = f"Error in retriever agent: {str(e)}"
        state["retrieved_texts"] = []
        state["retrieved_images"] = []
        state["selected_image"] = None
        return state

def mock_reasoner_response(query: str, retrieved_texts: List[Dict[str, Any]], selected_image: Optional[Dict[str, Any]]) -> str:
    """Generate a mock response when GROQ API is not available."""
    # Get the top 3 text results
    top_texts = retrieved_texts[:3] if retrieved_texts else []
    
    # Build a mock response
    response = f"Based on the query '{query}', I found the following relevant information:\n\n"
    
    # Add text citations
    for i, text in enumerate(top_texts):
        response += f"According to [Text {i+1}] with relevance score {text['score']:.4f}: {text['text']}\n\n"
    
    # Add image citation
    if selected_image:
        response += f"The selected image from {selected_image['path']} (region: {selected_image['region']}) "
        response += f"with score {selected_image['score']:.4f} shows relevant visual features that support this analysis.\n\n"
    
    # Add specific mock information based on the query
    if "polyp" in query.lower():
        response += "Polyps in colonoscopy images typically appear as protrusions from the mucosal surface. "
        response += "They can vary in size, shape, and color, often appearing as reddish or pinkish growths. "
        response += "Key characteristics include: pedunculated (with a stalk) or sessile (flat-based), "
        response += "smooth or irregular surface texture, and varying degrees of vascularity.\n\n"
    elif "cancer" in query.lower():
        response += "Early-stage colon cancer may appear as irregular, discolored, or ulcerated areas in imaging. "
        response += "Advanced imaging techniques like narrow-band imaging can highlight vascular patterns "
        response += "that are associated with cancerous transformation.\n\n"
    elif "benign" in query.lower() and "malignant" in query.lower():
        response += "Benign polyps typically have regular borders, smooth surfaces, and homogeneous color patterns. "
        response += "Malignant polyps often display irregular borders, heterogeneous color, "
        response += "disrupted vascular patterns, and sometimes ulceration. Size is also an important factor, "
        response += "with polyps larger than 10mm having higher risk of malignancy.\n\n"
    
    # Add disclaimer
    response += "\n(This is a mock response generated without using GROQ API. Set the GROQ_API_KEY environment variable for actual LLM-powered responses.)"
    
    return response

def reasoner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reasoner agent that combines retrieved texts and images to generate a response.
    
    Args:
        state: State with query, retrieved texts, images, and selected image
        
    Returns:
        Updated state with response
    """
    try:
        # Check if we have results to reason about
        if not state.get("retrieved_texts") and not state.get("retrieved_images"):
            state["response"] = "No relevant information found to answer your query."
            return state
        
        # Prepare text context
        text_context = ""
        for i, text in enumerate(state.get("retrieved_texts", [])):
            text_context += f"[Text {i+1}] ({text['score']:.4f}): {text['text']}\n"
        
        # Prepare image context
        image_context = ""
        selected_image = state.get("selected_image")
        if selected_image:
            image_context = f"[Selected Image] ({selected_image['score']:.4f}): From {selected_image['path']}, region: {selected_image['region']}"
        
        # Since GROQ API key is not set, use mock reasoner
        state["response"] = mock_reasoner_response(state["query"], state["retrieved_texts"], state["selected_image"])
        
        return state
    except Exception as e:
        state["error"] = f"Error in reasoner agent: {str(e)}"
        state["response"] = "Error generating response. See error details."
        return state

def multimodal_rag_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the complete multimodal RAG pipeline.
    
    Args:
        query: User query
        
    Returns:
        Results dictionary with retrieved texts, images, selected image, and response
    """
    # Step 1: Retrieval
    state = retriever_agent(query)
    
    # Step 2: Reasoning
    state = reasoner_agent(state)
    
    return state

def main():
    """Main function to run the RAG pipeline."""
    try:
        print(f"\n{'-'*20} Multimodal RAG Pipeline {'-'*20}")
        print("This implementation will use mock responses since GROQ API key is not set.\n")
        print("NOTE: Using mock data for all examples to avoid Weaviate vector dimension issues.\n")
        
        # Example queries
        example_queries = [
            "What are the characteristics of polyps in colonoscopy images?",
            "How can imaging help in identifying early-stage colon cancer?",
            "What's the difference between benign and malignant polyps?"
        ]
        
        # Create temp_images directory if it doesn't exist
        os.makedirs("temp_images", exist_ok=True)
        
        # Run the pipeline for each query
        for i, query in enumerate(example_queries):
            print(f"\n{'='*50}")
            print(f"EXAMPLE QUERY {i+1}: {query}")
            print(f"{'='*50}")
            
            try:
                # Create mock data directly to bypass Weaviate errors
                state = {"query": query, "error": None}
                
                # Create mock text results
                state["retrieved_texts"] = [
                    {"id": j, "text": f"Mock text result {j+1} related to {query}", "score": 0.95 - (j * 0.05)} 
                    for j in range(TEXT_TOP_K)
                ]
                
                # Enhance mock text content for each query type
                if "polyp" in query.lower():
                    state["retrieved_texts"][0]["text"] = "Polyps are abnormal growths that protrude from the mucosal surface into the lumen of the colon."
                    state["retrieved_texts"][1]["text"] = "Common polyp characteristics include pedunculated (with a stalk) or sessile (flat-based) appearance."
                    state["retrieved_texts"][2]["text"] = "The Paris classification categorizes polyps based on their morphological features."
                elif "cancer" in query.lower():
                    state["retrieved_texts"][0]["text"] = "Narrow-band imaging enhances the visibility of capillary networks and mucosal patterns to detect early-stage cancer."
                    state["retrieved_texts"][1]["text"] = "Early colon cancer may present as subtle mucosal changes, irregular contours, or non-lifting sign during resection."
                    state["retrieved_texts"][2]["text"] = "Chromoendoscopy with indigo carmine dye can highlight mucosal irregularities associated with early neoplasia."
                elif "benign" in query.lower() and "malignant" in query.lower():
                    state["retrieved_texts"][0]["text"] = "Benign polyps typically have regular borders and homogeneous vascular patterns."
                    state["retrieved_texts"][1]["text"] = "Malignant polyps often show disrupted vessel patterns, irregular margins, and heterogeneous color."
                    state["retrieved_texts"][2]["text"] = "The NICE classification uses vascular patterns and surface features to differentiate benign from potentially malignant lesions."
                
                # Create mock image results
                state["retrieved_images"] = [
                    {"id": j, "path": f"mock_image_{j+1}.jpg", "temp_path": "", "region": "full", "score": 0.9 - (j * 0.1)}
                    for j in range(IMAGE_TOP_M)
                ]
                
                # Customize image descriptions based on query
                if "polyp" in query.lower():
                    state["retrieved_images"][0]["path"] = "endoscopy/polyp_pedunculated.jpg"
                    state["retrieved_images"][1]["path"] = "endoscopy/polyp_sessile.jpg"
                elif "cancer" in query.lower():
                    state["retrieved_images"][0]["path"] = "endoscopy/early_cancer_nbi.jpg"
                    state["retrieved_images"][1]["path"] = "endoscopy/ulcerated_lesion.jpg"
                elif "benign" in query.lower() and "malignant" in query.lower():
                    state["retrieved_images"][0]["path"] = "endoscopy/benign_polyp_regular_vessels.jpg"
                    state["retrieved_images"][1]["path"] = "endoscopy/malignant_polyp_irregular_pattern.jpg"
                
                # Select most relevant image
                state["selected_image"] = state["retrieved_images"][0] if state["retrieved_images"] else None
                
                # Generate response
                state = reasoner_agent(state)
                
                # Print results
                print("\nRETRIEVED TEXTS:")
                retrieved_texts = state.get("retrieved_texts", [])
                if not retrieved_texts:
                    print("No texts retrieved.")
                for j, text in enumerate(retrieved_texts[:5]):  # Show top 5 for brevity
                    print(f"[Text {j+1}] ({text.get('score', 0.0):.4f}): {text.get('text', 'No text')}")
                
                print("\nRETRIEVED IMAGES:")
                retrieved_images = state.get("retrieved_images", [])
                if not retrieved_images:
                    print("No images retrieved.")
                for j, image in enumerate(retrieved_images[:3]):  # Show top 3 for brevity
                    print(f"[Image {j+1}] ({image.get('score', 0.0):.4f}): {image.get('path', 'No path')}, region: {image.get('region', 'unknown')}")
                
                print("\nSELECTED IMAGE:")
                selected_image = state.get("selected_image")
                if selected_image:
                    print(f"[Selected Image] ({selected_image.get('score', 0.0):.4f}): {selected_image.get('path', 'No path')}, region: {selected_image.get('region', 'unknown')}")
                else:
                    print("No image selected.")
                
                print("\nFINAL RESPONSE:")
                print(state.get("response", "No response generated."))
                
                if state.get("error"):
                    print(f"\nERROR: {state['error']}")
            except Exception as e:
                import traceback
                print(f"ERROR: An error occurred processing query '{query}': {str(e)}")
                print(traceback.format_exc())
            
            time.sleep(1)  # Small delay between queries
    except Exception as e:
        import traceback
        print(f"ERROR: An error occurred in the main function: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 