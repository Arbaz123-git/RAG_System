#!/usr/bin/env python
"""
MultiModal RAG with GROQ
This script implements a MultiModal RAG system for medical image analysis
using Weaviate as a vector database and GROQ as an LLM provider.
"""

import os
import sys
import time
import json
import random
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
from packaging import version
from colorama import init, Fore, Style

# Import dependencies with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_INSTALLED = True
except ImportError:
    DEPENDENCIES_INSTALLED = False

# Initialize colorama
init()

# Constants
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
TEXT_COLLECTION_NAME = "TextEmbeddings"
IMAGE_COLLECTION_NAME = "ImageEmbeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# GROQ API settings
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama3-70b-8192"  # Using LLaMA 3 70B model

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Print debugging info about the API key
print(f"GROQ_API_KEY found: {'Yes' if GROQ_API_KEY else 'No'}")
if GROQ_API_KEY:
    # Mask the API key for security
    masked_key = GROQ_API_KEY[:4] + "..." + GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 8 else "***"
    print(f"API Key (masked): {masked_key}")

def print_header(title, color=Fore.CYAN):
    """Print a styled header."""
    print("\n" + "=" * 50)
    print(color + title + Style.RESET_ALL)
    print("=" * 50)

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Dependencies", Fore.CYAN)
    
    if not DEPENDENCIES_INSTALLED:
        print("❌ Required packages are not installed.")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed.")
    return True

def check_weaviate_connection():
    """Check if Weaviate is running and accessible using direct REST API."""
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/meta")
        if response.status_code == 200:
            print("✅ Successfully connected to Weaviate")
            
            # Get schema to check if collections exist
            schema_response = requests.get(f"{WEAVIATE_URL}/v1/schema")
            if schema_response.status_code == 200:
                schema = schema_response.json()
                classes = [c["class"] for c in schema.get("classes", [])]
                
                if TEXT_COLLECTION_NAME in classes and IMAGE_COLLECTION_NAME in classes:
                    print("✅ Successfully accessed Weaviate schema")
                    return True
                else:
                    print("⚠️  Required collections not found in Weaviate schema.")
                    return False
            else:
                print("⚠️  Could not access Weaviate schema.")
                return False
        else:
            print(f"❌ Failed to connect to Weaviate: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        print("Make sure Weaviate is running (docker ps)")
        return False

def query_groq(prompt, max_tokens=1000):
    """Query the GROQ API with the given prompt."""
    if not GROQ_API_KEY:
        print("❌ GROQ API key not found in environment variables")
        return None
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides information about medical imaging and diagnoses based on the information retrieved from a knowledge base."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            f"{GROQ_API_BASE}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"❌ Failed to query GROQ API: {response.status_code}")
            print(response.text)
            return None
    
    except Exception as e:
        print(f"❌ Error querying GROQ API: {e}")
        return None

def retrieve_information_rest_api(query):
    """Retrieve relevant information from Weaviate using REST API."""
    try:
        # Load the embedding model
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        query_embedding = model.encode(query).tolist()
        
        # Query Weaviate for text information using GraphQL
        text_query = {
            "query": """
            {
              Get {
                TextEmbeddings(nearVector: {vector: %s, certainty: 0.7, limit: 5}) {
                  embedding_id
                  text
                  _additional {
                    certainty
                  }
                }
              }
            }
            """ % json.dumps(query_embedding)
        }
        
        text_response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json=text_query
        )
        
        # Extract text results
        text_data = []
        if text_response.status_code == 200:
            text_results = text_response.json()
            if 'data' in text_results and 'Get' in text_results['data'] and 'TextEmbeddings' in text_results['data']['Get']:
                text_objects = text_results['data']['Get']['TextEmbeddings']
                for obj in text_objects:
                    if 'text' in obj:
                        text_data.append(obj['text'])
        
        # Query Weaviate for image information using GraphQL
        image_query = {
            "query": """
            {
              Get {
                ImageEmbeddings(nearVector: {vector: %s, certainty: 0.7, limit: 3}) {
                  embedding_id
                  image_path
                  region
                  _additional {
                    certainty
                  }
                }
              }
            }
            """ % json.dumps(query_embedding)
        }
        
        image_response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json=image_query
        )
        
        # Extract image results
        image_data = []
        if image_response.status_code == 200:
            image_results = image_response.json()
            if 'data' in image_results and 'Get' in image_results['data'] and 'ImageEmbeddings' in image_results['data']['Get']:
                image_objects = image_results['data']['Get']['ImageEmbeddings']
                for obj in image_objects:
                    image_data.append({
                        'image_path': obj.get('image_path', ''),
                        'region': obj.get('region', '')
                    })
        
        # Combine the results
        return {
            'text_data': text_data,
            'image_data': image_data
        }
    
    except Exception as e:
        print(f"❌ Error retrieving information: {e}")
        return mock_retrieve_information(query)

def retrieve_information(query, use_weaviate=False):
    """Retrieve relevant information from Weaviate or mock data."""
    if use_weaviate:
        # Use direct REST API approach
        return retrieve_information_rest_api(query)
    else:
        # Return mock data when Weaviate is not available
        return mock_retrieve_information(query)

def mock_retrieve_information(query):
    """Generate mock retrieval data for demonstration purposes."""
    print("Using mock data for demonstration...")
    
    # Mock text data about colonoscopy and related medical information
    text_data = [
        "Polyps in colonoscopy images often appear as raised, rounded structures protruding from the intestinal wall. They vary in size from a few millimeters to several centimeters. The texture may be smooth or irregular, and the color ranges from pale to reddish, depending on vascularity.",
        "Early-stage colon cancer may present as subtle mucosal changes, irregular contours, or non-lifting sign during resection. Narrow-band imaging enhances the visibility of capillary networks and mucosal patterns to detect early-stage cancer. Chromoendoscopy with indigo carmine dye can highlight mucosal irregularities associated with early neoplasia.",
        "Benign polyps typically have a smooth surface with regular borders and uniform vascularity. Malignant polyps often show irregular borders, depression, abnormal surface patterns, and disrupted vascularity. The 'pit pattern' classification (Kudo classification) helps distinguish benign from malignant lesions based on crypt architecture.",
        "In colonoscopy images, diverticula appear as pocket-like recessions in the colon wall. They typically have a dark, shadowed opening and may contain debris. Diverticulosis is characterized by multiple diverticula scattered throughout the colon, while diverticulitis shows inflammation with redness and exudate around the diverticular openings.",
        "Inflammatory bowel disease (IBD) in colonoscopy shows mucosal erythema, edema, loss of vascular pattern, ulcerations, and in severe cases, pseudopolyps. Crohn's disease typically presents with skip lesions and deep, serpiginous ulcers, while ulcerative colitis shows continuous inflammation starting from the rectum with a granular, friable mucosa."
    ]
    
    # Mock image data descriptions
    image_data = [
        {
            "image_path": "images/colonoscopy_polyp_01.jpg",
            "region": "Sessile polyp in sigmoid colon, 8mm in diameter with smooth surface"
        },
        {
            "image_path": "images/colonoscopy_early_cancer_02.jpg",
            "region": "Early adenocarcinoma in ascending colon with irregular borders and depression"
        },
        {
            "image_path": "images/colonoscopy_normal_03.jpg",
            "region": "Normal colonic mucosa with visible vascular pattern and haustra"
        }
    ]
    
    # Return mock data
    return {
        'text_data': random.sample(text_data, min(3, len(text_data))),
        'image_data': random.sample(image_data, min(2, len(image_data)))
    }

def generate_response(query, retrieved_info):
    """Generate a response based on the query and retrieved information."""
    # If GROQ API key is available, use it to generate a response
    if GROQ_API_KEY:
        # Construct a prompt for the GROQ API
        prompt = f"""
        Question: {query}
        
        Retrieved information from the database:
        
        Text Data:
        {'-' * 40}
        {('\n' + '-' * 40 + '\n').join(retrieved_info['text_data'])}
        
        Image Data:
        {'-' * 40}
        {('\n' + '-' * 40 + '\n').join([f"Image: {img['image_path']}\nDescription: {img['region']}" for img in retrieved_info['image_data']])}
        
        Based on the retrieved information, provide a comprehensive answer to the question.
        Focus on the medical imaging aspects and be specific about what can be observed in the images.
        """
        
        # Query GROQ API
        print("Querying GROQ API...")
        response = query_groq(prompt)
        
        if response:
            return response
        else:
            print("⚠️  Falling back to mock response due to GROQ API error")
    else:
        print("⚠️  GROQ API key not found, using mock response")
    
    # Fall back to mock response if GROQ is not available
    return mock_generate_response(query, retrieved_info)

def mock_generate_response(query, retrieved_info):
    """Generate a mock response when GROQ API is not available."""
    # Sample responses for different query types
    responses = {
        "polyp": "Based on the retrieved information:\n\nPolyps in colonoscopy images appear as raised, rounded structures protruding from the intestinal wall. Their size ranges from a few millimeters to several centimeters, with textures varying from smooth to irregular. The color can range from pale to reddish, depending on the blood vessel content.\n\nThe images show examples of both sessile polyps (flat, broad-based) and pedunculated polyps (attached by a stalk). Benign polyps typically have a smooth surface with regular borders, while potentially malignant ones show irregular borders and abnormal surface patterns.",
        "cancer": "Based on the retrieved information:\n\nEarly-stage colon cancer in imaging can be identified through several visual indicators. These include subtle mucosal changes, irregular contours, and the non-lifting sign during attempted resection.\n\nAdvanced imaging techniques enhance detection capabilities: narrow-band imaging highlights capillary networks and mucosal patterns, while chromoendoscopy with indigo carmine dye accentuates mucosal irregularities associated with early neoplasia.\n\nThe images show examples of early adenocarcinoma with characteristics like irregular borders, surface depression, and disrupted vascular patterns.",
        "difference": "Based on the retrieved information:\n\nThe key differences between benign and malignant polyps in imaging are:\n\n1. Surface appearance: Benign polyps have smooth surfaces with regular borders, while malignant polyps show irregular, rough surfaces\n\n2. Borders: Benign polyps have well-defined, regular borders; malignant polyps display irregular, poorly defined borders\n\n3. Vascularity: Benign polyps show uniform vascularity patterns, while malignant polyps exhibit disrupted, abnormal vessel patterns\n\n4. Depression: Malignant polyps often show central depression or ulceration\n\n5. Pit pattern: Using the Kudo classification, the crypt architecture differs between benign and malignant lesions",
        "diverticula": "Based on the retrieved information:\n\nIn colonoscopy images, diverticula appear as pocket-like recessions or outpouchings in the colon wall. They typically present with dark, shadowed openings and may contain debris or stool material.\n\nDiverticulosis is characterized by the presence of multiple diverticula scattered throughout the colon, most commonly in the sigmoid and descending colon. The surrounding mucosa may appear normal in uncomplicated cases.\n\nDiverticulitis shows signs of inflammation around the diverticular openings, including redness, edema, and possible exudate. In severe cases, there may be visible pus, narrowing of the lumen, or fistula formation.",
        "inflammatory": "Based on the retrieved information:\n\nInflammatory bowel disease (IBD) presents with distinctive features in colonoscopy images. Common findings include mucosal erythema (redness), edema, loss of the normal vascular pattern, and various degrees of ulceration.\n\nCrohn's disease typically shows a discontinuous pattern with 'skip lesions' (affected areas separated by normal mucosa). Deep, serpiginous (snake-like) ulcers are characteristic, and there may be a 'cobblestone' appearance in severely affected areas.\n\nUlcerative colitis, in contrast, presents with continuous inflammation that typically starts in the rectum and extends proximally. The mucosa appears granular, friable (easily bleeding), and may show pseudopolyps in chronic cases. The demarcation between inflamed and normal mucosa is usually clear."
    }
    
    # Determine which response to use based on keywords in the query
    query_lower = query.lower()
    if "polyp" in query_lower or "growth" in query_lower:
        return responses["polyp"]
    elif "cancer" in query_lower or "malignant" in query_lower or "tumor" in query_lower:
        return responses["cancer"]
    elif "difference" in query_lower or "distinguish" in query_lower or "benign" in query_lower:
        return responses["difference"]
    elif "diverticula" in query_lower or "diverticulosis" in query_lower or "diverticulitis" in query_lower:
        return responses["diverticula"]
    elif "inflammatory" in query_lower or "inflammation" in query_lower or "crohn" in query_lower or "colitis" in query_lower:
        return responses["inflammatory"]
    else:
        # Generic response for other queries
        return "Based on the retrieved information:\n\nMedical imaging plays a crucial role in identifying various conditions in the gastrointestinal tract. Colonoscopy images can reveal polyps, early-stage cancer, diverticula, inflammatory conditions, and other abnormalities.\n\nThe interpretation of these images relies on careful observation of features such as surface texture, color, vascularity patterns, and surrounding tissue appearance. Advanced imaging techniques like narrow-band imaging and chromoendoscopy enhance the visibility of these features.\n\nEarly detection of abnormalities through imaging significantly improves treatment outcomes, particularly for conditions like colorectal cancer where early intervention is critical."

def interactive_demo():
    """Run an interactive demo of the MultiModal RAG system."""
    print_header("MultiModal RAG System", Fore.GREEN)
    
    # Check if Weaviate is available
    use_weaviate = check_weaviate_connection()
    
    if not use_weaviate:
        print("This is a mock version of the system using hardcoded responses.")
    else:
        print("This system retrieves information from Weaviate and generates responses using GROQ.")
    
    # Example queries
    examples = [
        "What are the characteristics of polyps in colonoscopy images?",
        "How can imaging help in identifying early-stage colon cancer?",
        "What's the difference between benign and malignant polyps?",
        "How do diverticula appear in colonoscopy images?",
        "What are the imaging features of inflammatory bowel disease?"
    ]
    
    # Print example queries
    print("\nExample queries:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    # Main interaction loop
    while True:
        print("\nEnter a number to select an example query, or type your own query: ", end="")
        user_input = input().strip()
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting demo...")
            break
        
        # Process user input
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            query = examples[int(user_input) - 1]
        else:
            query = user_input
        
        print(f"\nProcessing query: '{query}'")
        
        # Retrieve information
        print("Retrieving relevant information...")
        retrieved_info = retrieve_information(query, use_weaviate)
        
        # Generate response
        print("\nGenerating response...")
        response = generate_response(query, retrieved_info)
        
        # Display response
        print_header("Response", Fore.GREEN)
        print(response)

def main():
    """Main function."""
    # Check dependencies
    if not check_dependencies():
        print_header("Mock MultiModal RAG System", Fore.YELLOW)
        interactive_demo()
        return
    
    # Run interactive demo
    interactive_demo()

if __name__ == "__main__":
    main()