#!/usr/bin/env python
# Multimodal RAG Agent using LangGraph
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import time
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Same model used in the benchmark
TEXT_TOP_K = 10  # Number of text results to retrieve
IMAGE_TOP_M = 5  # Number of image results to retrieve
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set your GROQ API key as environment variable

# Check if GROQ API key is set
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable not set. LLM-based reasoning will not work.")

# Define state as a dictionary
class AgentState(Dict[str, Any]):
    """State for the RAG agent workflow using a dictionary to be compatible with LangGraph."""
    @property
    def query(self) -> str:
        return self.get("query", "")
    
    @property
    def retrieved_texts(self) -> List[Dict[str, Any]]:
        return self.get("retrieved_texts", [])
    
    @property
    def retrieved_images(self) -> List[Dict[str, Any]]:
        return self.get("retrieved_images", [])
    
    @property
    def selected_image(self) -> Optional[Dict[str, Any]]:
        return self.get("selected_image", None)
    
    @property
    def response(self) -> Optional[str]:
        return self.get("response", None)
    
    @property
    def error(self) -> Optional[str]:
        return self.get("error", None)

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

# Agent components
def retriever_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retriever agent that embeds the query and retrieves relevant text and images.
    """
    try:
        # Set up Weaviate client and sentence transformer
        try:
            client = setup_weaviate_client()
        except Exception as e:
            print(f"Warning: Failed to connect to Weaviate: {e}")
            print("Using mock data for demonstration purposes.")
            
            # Create mock data for demonstration
            state["retrieved_texts"] = [
                {"id": i, "text": f"Mock text result {i+1} for query: {state['query']}", "score": 0.95 - (i * 0.05)} 
                for i in range(TEXT_TOP_K)
            ]
            state["retrieved_images"] = [
                {"id": i, "path": f"mock_image_{i+1}.jpg", "temp_path": "", "region": "full", "score": 0.9 - (i * 0.1)}
                for i in range(IMAGE_TOP_M)
            ]
            state["selected_image"] = state["retrieved_images"][0] if state["retrieved_images"] else None
            
            return state
        
        model = load_sentence_transformer()
        
        # Embed the query
        query_embedding = model.encode(state["query"])
        
        # Retrieve relevant text
        text_collection = client.collections.get(TEXT_COLLECTION_NAME)
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
        
        # Select the most relevant image (highest score)
        selected_image = retrieved_images[0] if retrieved_images else None
        
        # Update state
        state["retrieved_texts"] = retrieved_texts
        state["retrieved_images"] = retrieved_images
        state["selected_image"] = selected_image
        
        client.close()
        return state
    except Exception as e:
        state["error"] = f"Error in retriever agent: {str(e)}"
        return state

def reasoner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reasoner agent that combines retrieved texts and images to generate a response.
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
        
        # Check if GROQ API key is available, otherwise use mock reasoner
        if not GROQ_API_KEY:
            state["response"] = mock_reasoner_response(state["query"], state["retrieved_texts"], state["selected_image"])
            return state
        
        # Create prompt for LLM
        prompt_template = """
        You are a medical research assistant with expertise in analyzing medical images and research papers.
        
        USER QUERY: {query}
        
        RETRIEVED TEXT CONTEXTS:
        {text_context}
        
        RETRIEVED IMAGE CONTEXT:
        {image_context}
        
        Please provide a concise answer that:
        1. Directly quotes the most relevant sentences from the text contexts (using their identifiers like [Text 1])
        2. Explains why the selected image is relevant to the query
        3. Cites all sources used in your answer
        
        Your answer should be informative, accurate, and focused on answering the user's query using only the provided contexts.
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Initialize LLM
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-70b-8192"  # Using Llama 3 70B model
        )
        
        # Generate response
        chain = prompt | llm
        response = chain.invoke({
            "query": state["query"],
            "text_context": text_context,
            "image_context": image_context
        })
        
        # Update state with the response
        state["response"] = response.content if GROQ_API_KEY else "GROQ API key not set. Here's a mock response based on retrieved data."
        
        return state
    except Exception as e:
        state["error"] = f"Error in reasoner agent: {str(e)}"
        if not GROQ_API_KEY:
            state["response"] = "GROQ API key not set. Unable to generate a response using the LLM."
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
    
    # Add disclaimer
    response += "\n(This is a mock response generated without using GROQ API. Set the GROQ_API_KEY environment variable for actual LLM-powered responses.)"
    
    return response

# Define the workflow
def create_graph() -> StateGraph:
    """Create the LangGraph workflow."""
    # Create a new graph
    graph = StateGraph(AgentState)
    
    # Add nodes for each agent
    graph.add_node("retriever", retriever_agent)
    graph.add_node("reasoner", reasoner_agent)
    
    # Define the edges
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", END)
    
    # Set the entry point
    graph.set_entry_point("retriever")
    
    # Compile the graph
    return graph.compile()

# Main execution
def main():
    """Main function to run the RAG agent."""
    try:
        # Create the graph
        graph = create_graph()
        
        # Example queries
        example_queries = [
            "What are the characteristics of polyps in colonoscopy images?",
            "How can imaging help in identifying early-stage colon cancer?",
            "What's the difference between benign and malignant polyps?"
        ]
        
        # Create temp_images directory if it doesn't exist
        os.makedirs("temp_images", exist_ok=True)
        
        # Run the agent for each query
        for i, query in enumerate(example_queries):
            print(f"\n{'='*50}")
            print(f"EXAMPLE QUERY {i+1}: {query}")
            print(f"{'='*50}")
            
            try:
                # Run the graph with a timeout to prevent hanging
                initial_state = {"query": query}
                result = graph.invoke(initial_state)
                
                # Ensure result is not None and has the expected structure
                if result is None:
                    print("WARNING: Graph returned None instead of a state object.")
                    result = initial_state
                
                # Print results
                print("\nRETRIEVED TEXTS:")
                retrieved_texts = result.get("retrieved_texts", [])
                if not retrieved_texts:
                    print("No texts retrieved.")
                for j, text in enumerate(retrieved_texts):
                    print(f"[Text {j+1}] ({text.get('score', 0.0):.4f}): {text.get('text', 'No text')}")
                
                print("\nRETRIEVED IMAGES:")
                retrieved_images = result.get("retrieved_images", [])
                if not retrieved_images:
                    print("No images retrieved.")
                for j, image in enumerate(retrieved_images):
                    print(f"[Image {j+1}] ({image.get('score', 0.0):.4f}): {image.get('path', 'No path')}, region: {image.get('region', 'unknown')}")
                
                print("\nSELECTED IMAGE:")
                selected_image = result.get("selected_image")
                if selected_image:
                    print(f"[Selected Image] ({selected_image.get('score', 0.0):.4f}): {selected_image.get('path', 'No path')}, region: {selected_image.get('region', 'unknown')}")
                else:
                    print("No image selected.")
                
                print("\nFINAL RESPONSE:")
                print(result.get("response", "No response generated."))
                
                if result.get("error"):
                    print(f"\nERROR: {result['error']}")
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