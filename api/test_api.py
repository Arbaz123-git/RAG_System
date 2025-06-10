#!/usr/bin/env python
"""
Test script for the MultiModal RAG API
This script demonstrates how to use the API endpoints.
"""

import requests
import json
import sys
import os

# Constants
API_URL = "http://localhost:8000"
USERNAME = "clinician1"
PASSWORD = "secret1"

def get_token():
    """Get a JWT token from the API."""
    print("Getting JWT token...")
    
    response = requests.post(
        f"{API_URL}/token",
        data={
            "username": USERNAME,
            "password": PASSWORD
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }
    )
    
    if response.status_code == 200:
        token_data = response.json()
        print("✅ Successfully got JWT token")
        return token_data["access_token"]
    else:
        print(f"❌ Failed to get JWT token: {response.status_code}")
        print(response.text)
        return None

def ask_question(token, query):
    """Ask a question to the MultiModal RAG API."""
    print(f"\nAsking question: {query}")
    
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "query": query
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Successfully got answer")
        return result
    else:
        print(f"❌ Failed to get answer: {response.status_code}")
        print(response.text)
        return None

def print_result(result):
    """Print the result in a formatted way."""
    if not result:
        return
    
    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(result["answer"])
    
    print("\n" + "=" * 50)
    print("METADATA")
    print("=" * 50)
    print(f"Timestamp: {result['metadata']['timestamp']}")
    print(f"User: {result['metadata']['user']}")
    
    if "sources" in result["metadata"] and result["metadata"]["sources"]:
        print("\nSources:")
        for i, source in enumerate(result["metadata"]["sources"], 1):
            print(f"{i}. {source}")
    
    if "image_sources" in result["metadata"] and result["metadata"]["image_sources"]:
        print("\nImage Sources:")
        for i, source in enumerate(result["metadata"]["image_sources"], 1):
            print(f"{i}. {source}")

def main():
    """Main function."""
    print("=" * 50)
    print("MultiModal RAG API Test")
    print("=" * 50)
    
    # Get token
    token = get_token()
    if not token:
        print("Exiting due to authentication error")
        return
    
    # Example queries
    queries = [
        "What are the characteristics of polyps in colonoscopy images?",
        "How can imaging help in identifying early-stage colon cancer?",
        "What's the difference between benign and malignant polyps?"
    ]
    
    # Ask each query
    for query in queries:
        result = ask_question(token, query)
        if result:
            print_result(result)
    
    print("\n" + "=" * 50)
    print("Test completed")
    print("=" * 50)

if __name__ == "__main__":
    main() 