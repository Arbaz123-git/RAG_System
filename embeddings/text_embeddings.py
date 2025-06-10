import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os
import argparse

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_data(csv_path, num_records=200000):
    """
    Load the PubMed abstracts from CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    num_records : int
        Number of records to load
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the abstracts
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Take only the specified number of records
    df = df.head(num_records)
    
    print(f"Loaded {len(df)} records")
    return df

def split_abstracts_into_sentences(abstracts):
    """
    Split abstracts into sentences
    
    Parameters:
    -----------
    abstracts : list
        List of abstract texts
        
    Returns:
    --------
    list
        List of sentences
    list
        List of sentence-to-abstract mappings
    """
    all_sentences = []
    sentence_to_abstract_mapping = []
    
    print("Splitting abstracts into sentences...")
    for i, abstract in enumerate(tqdm(abstracts)):
        # Skip empty or NaN abstracts
        if not isinstance(abstract, str) or not abstract.strip():
            continue
            
        # Split abstract into sentences
        sentences = sent_tokenize(abstract)
        
        # Add sentences to the list
        all_sentences.extend(sentences)
        
        # Keep track of which abstract each sentence belongs to
        sentence_to_abstract_mapping.extend([i] * len(sentences))
    
    print(f"Total sentences: {len(all_sentences)}")
    return all_sentences, sentence_to_abstract_mapping

def embed_sentences(sentences, model_name="all-mpnet-base-v2", batch_size=32, device=None):
    """
    Embed sentences using SentenceTransformers
    
    Parameters:
    -----------
    sentences : list
        List of sentences to embed
    model_name : str
        Name of the SentenceTransformers model to use
    batch_size : int
        Batch size for embedding
    device : str
        Device to use for embedding (None for auto-detection)
        
    Returns:
    --------
    numpy.ndarray
        Array of sentence embeddings
    """
    print(f"Loading SentenceTransformers model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    print("Generating embeddings...")
    embeddings = model.encode(
        sentences, 
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def save_embeddings(embeddings, output_path):
    """
    Save embeddings to a file
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Array of embeddings
    output_path : str
        Path to save the embeddings
    """
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    print("Embeddings saved successfully")

def main():
    parser = argparse.ArgumentParser(description='Generate text embeddings from PubMed abstracts')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--output_path', type=str, default='sentence_embeddings.npy', help='Path to save the embeddings')
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', help='SentenceTransformers model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding')
    parser.add_argument('--num_records', type=int, default=10000, help='Number of records to process')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda:0, etc.)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Load data
    df = load_data(args.csv_path, args.num_records)
    
    # Extract abstracts
    abstracts = df['abstract_text'].tolist()
    
    # Split abstracts into sentences
    sentences, sentence_to_abstract_mapping = split_abstracts_into_sentences(abstracts)
    
    # Generate embeddings
    embeddings = embed_sentences(sentences, args.model_name, args.batch_size, args.device)
    
    # Save embeddings
    save_embeddings(embeddings, args.output_path)
    
    # Save sentence-to-abstract mapping
    mapping_path = args.output_path.replace('.npy', '_mapping.npy')
    np.save(mapping_path, np.array(sentence_to_abstract_mapping))
    print(f"Sentence-to-abstract mapping saved to {mapping_path}")
    
    # Save sentences for reference
    sentences_path = args.output_path.replace('.npy', '_sentences.txt')
    with open(sentences_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    print(f"Sentences saved to {sentences_path}")
    
    print("Done!")

if __name__ == "__main__":
    main() 

# python embeddings/text_embeddings.py --csv_path data/train.csv/train.csv --output_path output/sentence_embeddings.npy --num_records 10000
