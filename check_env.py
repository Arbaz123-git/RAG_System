#!/usr/bin/env python
"""
Simple script to check if the .env file is being loaded correctly.
"""

import os
from dotenv import load_dotenv

def main():
    """Check if the .env file is being loaded correctly."""
    print("Checking .env file...")
    
    # First, try to load the .env file
    load_dotenv()
    
    # Check if GROQ_API_KEY is in environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        # Mask the API key for security
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"✅ GROQ_API_KEY found in environment variables: {masked_key}")
    else:
        print("❌ GROQ_API_KEY not found in environment variables")
    
    # Check if the .env file exists
    if os.path.exists(".env"):
        print("✅ .env file exists in the current directory")
        
        # Read the file directly to verify its contents
        try:
            with open(".env", "r") as f:
                contents = f.read()
                if "GROQ_API_KEY" in contents:
                    print("✅ GROQ_API_KEY found in .env file")
                else:
                    print("❌ GROQ_API_KEY not found in .env file contents")
                
                # Print the file contents with API key masked
                masked_contents = contents
                if "GROQ_API_KEY" in contents:
                    # Find the API key in the file
                    lines = contents.split("\n")
                    for i, line in enumerate(lines):
                        if "GROQ_API_KEY" in line:
                            # Mask the API key
                            parts = line.split("=", 1)
                            if len(parts) > 1:
                                key = parts[1].strip()
                                if key:
                                    masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
                                    lines[i] = f"{parts[0]}= {masked_key}"
                    
                    masked_contents = "\n".join(lines)
                
                print("\n.env file contents (with API key masked):")
                print("-" * 40)
                print(masked_contents)
                print("-" * 40)
        
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print("❌ .env file does not exist in the current directory")
    
    # Check if there's an absolute path issue
    print("\nCurrent working directory:", os.getcwd())
    
    # Look for .env in parent directories
    parent_dir = os.path.dirname(os.getcwd())
    if os.path.exists(os.path.join(parent_dir, ".env")):
        print(f"✅ .env file found in parent directory: {parent_dir}")
    
    # Additional debug information
    print("\nEnvironment variables related to GROQ:")
    for key, value in os.environ.items():
        if "GROQ" in key.upper():
            masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"{key}: {masked_value}")

if __name__ == "__main__":
    main()
