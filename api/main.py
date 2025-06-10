#!/usr/bin/env python
"""
Secure REST API for MultiModal RAG
This module provides a secure REST API with JWT authentication for the MultiModal RAG system.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import jwt
from fastapi import FastAPI, Depends, HTTPException, status, Security, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to import MultiModal RAG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MultiModal RAG functions
from multimodal_rag_with_groq import retrieve_information_rest_api, generate_response

# Import the two-tier cache
from cache import TwoTierCache

# Import cache monitoring
from cache_monitor import cache_monitor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# JWT Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-development-only")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60

# Cache Settings
MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "5000"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))  # 24 hours in seconds

# Initialize the two-tier cache
cache = TwoTierCache(
    memory_cache_size=MEMORY_CACHE_SIZE,
    redis_host=REDIS_HOST,
    redis_port=REDIS_PORT,
    redis_db=REDIS_DB,
    redis_password=REDIS_PASSWORD,
    redis_ttl=REDIS_TTL
)

# Security
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create FastAPI app
app = FastAPI(
    title="MultiModal RAG API",
    description="A secure REST API for MultiModal RAG with JWT authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class AskRequest(BaseModel):
    query: str
    token: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]
    cache_hit: Optional[str] = None

# Mock user database - In production, use a real database
fake_users_db = {
    "clinician1": {
        "username": "clinician1",
        "hashed_password": "fakehashedsecret1",
        "disabled": False,
    },
    "clinician2": {
        "username": "clinician2",
        "hashed_password": "fakehashedsecret2",
        "disabled": False,
    },
}

# Authentication functions
def verify_password(plain_password, hashed_password):
    # In production, use a proper password hashing library like bcrypt
    return plain_password == hashed_password.replace("fakehashed", "")

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    token_payload: Dict = Depends(validate_token)
):
    # Log request (in production, use proper logging)
    print(f"Received query: {request.query}")
    print(f"From user: {token_payload.get('sub')}")
    
    try:
        # Start timing for latency measurement
        start_time = time.time()
        
        # Use the query string as the cache key
        cache_key = request.query
        
        # Check cache first
        cached_result, cache_source = cache.get(cache_key)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        if cached_result:
            print(f"Cache hit from {cache_source}")
            # Add cache hit source to metadata
            cached_result["metadata"]["cache_hit"] = cache_source
            
            # Record cache hit in monitor
            cache_monitor.record_cache_access(cache_source, latency_ms)
            
            return cached_result
        
        # Cache miss, perform the actual query
        print("Cache miss, retrieving information...")
        
        # Record the start time for RAG processing
        rag_start_time = time.time()
        
        # Retrieve information from Weaviate
        retrieved_info = retrieve_information_rest_api(request.query)
        
        # Generate response using GROQ
        response = generate_response(request.query, retrieved_info)
        
        # Calculate RAG processing latency
        rag_latency_ms = (time.time() - rag_start_time) * 1000
        
        # Record cache miss in monitor
        cache_monitor.record_cache_access("miss", rag_latency_ms)
        
        # Prepare metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": token_payload.get("sub"),
            "sources": [],
            "cache_hit": "miss"
        }
        
        # Add sources to metadata if available
        if retrieved_info and "text_data" in retrieved_info:
            # Add first 100 characters of each text source
            metadata["sources"] = [text[:100] + "..." for text in retrieved_info.get("text_data", [])]
        
        # Add image sources if available
        if retrieved_info and "image_data" in retrieved_info:
            metadata["image_sources"] = [img.get("image_path", "") for img in retrieved_info.get("image_data", [])]
        
        # Create the response object
        result = {
            "answer": response,
            "metadata": metadata
        }
        
        # Store in cache
        cache.set(cache_key, result)
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "MultiModal RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/token": "Get JWT token (POST)",
            "/ask": "Ask a question (POST)"
        }
    }

@app.get("/cache/stats")
async def cache_stats(current_user: User = Depends(get_current_active_user)):
    """Get current cache statistics."""
    # Save current stats to file
    cache_monitor.save_stats()
    
    # Return a summary of cache statistics
    total = (cache_monitor.stats["miss"] + 
             cache_monitor.stats["memory"] + 
             cache_monitor.stats["redis"] + 
             cache_monitor.stats["unknown"])
    
    if total == 0:
        hit_rate = 0
    else:
        hit_rate = ((cache_monitor.stats["memory"] + cache_monitor.stats["redis"]) / total) * 100
    
    return {
        "total_requests": total,
        "cache_misses": cache_monitor.stats["miss"],
        "memory_hits": cache_monitor.stats["memory"],
        "redis_hits": cache_monitor.stats["redis"],
        "unknown": cache_monitor.stats["unknown"],
        "hit_rate": hit_rate
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 