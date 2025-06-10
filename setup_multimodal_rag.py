#!/usr/bin/env python
"""
Simple setup script for the MultiModal RAG System
"""
import os
import sys
import subprocess
import platform

# Required packages for the MultiModal RAG system
REQUIRED_PACKAGES = [
    "weaviate-client>=4.5.0",
    "sentence-transformers>=2.2.2",
    "numpy>=1.23.0",
    "pillow>=9.0.0",
    "matplotlib>=3.5.0",
    "langchain>=0.1.0",
    "langchain-groq>=0.1.0",
    "requests>=2.28.0",
    "python-dotenv>=1.0.0"
]

def print_header(message):
    """Print a header message with decoration."""
    print("\n" + "=" * 50)
    print(f"{message}")
    print("=" * 50 + "\n")

def create_requirements_file():
    """Create a requirements.txt file."""
    print("Creating requirements.txt file...")
    with open("requirements.txt", "w") as f:
        f.write("\n".join(REQUIRED_PACKAGES))
    print("✅ Created requirements.txt file")

def check_docker():
    """Check if Docker is installed and running."""
    print("\nChecking Docker installation...")
    
    try:
        subprocess.check_output(["docker", "--version"])
        print("✅ Docker is installed.")
        
        # Check if Docker is running
        subprocess.check_output(["docker", "ps"])
        print("✅ Docker is running.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ Docker is not installed or not running.")
        print("Please install Docker and make sure it's running:")
        print("- Download from: https://www.docker.com/products/docker-desktop")
        print("- Start Docker Desktop")
        return False

def check_weaviate():
    """Check if Weaviate is running."""
    print("\nChecking Weaviate container...")
    
    try:
        output = subprocess.check_output(["docker", "ps"], text=True)
        if "weaviate" in output:
            print("✅ Weaviate container is running.")
            return True
        else:
            print("❌ Weaviate container is not running.")
            return False
    except subprocess.CalledProcessError:
        print("❌ Failed to check Docker containers.")
        return False

def provide_next_steps():
    """Provide next steps for the user."""
    print_header("Next Steps")
    
    print("To complete the setup of your MultiModal RAG system:")
    
    print("\n1. Install the required packages:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Set up Weaviate:")
    print("   docker run -d --name weaviate -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:1.24.1")
    
    print("\n3. Get a GROQ API key:")
    print("   Sign up at https://console.groq.com/ and get your API key")
    
    print("\n4. Create the Weaviate schema:")
    print("   python create_weaviate_schema.py")
    
    print("\n5. Upload embeddings to Weaviate:")
    print("   python upload_embeddings.py")
    
    print("\n6. Run the MultiModal RAG system:")
    print("   python multimodal_rag_with_groq.py")

def main():
    """Main function to set up the MultiModal RAG system."""
    print_header("MultiModal RAG System Setup")
    print("This script will help you set up the necessary components for the MultiModal RAG system.")
    
    # Create requirements.txt file
    create_requirements_file()
    
    # Check Docker and Weaviate
    docker_running = check_docker()
    if docker_running:
        weaviate_running = check_weaviate()
    
    # Provide next steps
    provide_next_steps()
    
    print("\nSetup script completed!")

if __name__ == "__main__":
    main()