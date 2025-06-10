import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate text and image embeddings')
    
    # General arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda:0, etc.)')
    
    # Text embedding arguments
    parser.add_argument('--csv_path', type=str, default='data/train.csv/train.csv', help='Path to the CSV file')
    parser.add_argument('--text_model', type=str, default='all-mpnet-base-v2', help='SentenceTransformers model for text')
    parser.add_argument('--num_records', type=int, default=10000, help='Number of text records to process')
    
    # Image embedding arguments
    parser.add_argument('--image_dir', type=str, default='data/kvasir-seg/Kvasir-SEG/images', help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, default='data/kvasir-seg/Kvasir-SEG/masks', help='Directory containing masks')
    parser.add_argument('--image_model', type=str, default='clip-ViT-B-32', help='SentenceTransformers model for images')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of image patches')
    parser.add_argument('--image_limit', type=int, default=100, help='Number of images to process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process text embeddings
    print("=== Generating Text Embeddings ===")
    text_output_path = os.path.join(args.output_dir, 'sentence_embeddings.npy')
    
    text_cmd = [
        sys.executable, 'embeddings/text_embeddings.py',
        '--csv_path', args.csv_path,
        '--output_path', text_output_path,
        '--model_name', args.text_model,
        '--num_records', str(args.num_records)
    ]
    
    if args.device:
        text_cmd.extend(['--device', args.device])
    
    print(f"Running command: {' '.join(text_cmd)}")
    subprocess.run(text_cmd)
    
    # Process image embeddings
    print("\n=== Generating Image Embeddings ===")
    image_output_path = os.path.join(args.output_dir, 'image_embeddings.npy')
    
    image_cmd = [
        sys.executable, 'embeddings/image_embeddings.py',
        '--image_dir', args.image_dir,
        '--mask_dir', args.mask_dir,
        '--output_path', image_output_path,
        '--model_name', args.image_model,
        '--patch_size', str(args.patch_size),
        '--limit', str(args.image_limit)
    ]
    
    if args.device:
        image_cmd.extend(['--device', args.device])
    
    print(f"Running command: {' '.join(image_cmd)}")
    subprocess.run(image_cmd)
    
    print("\n=== Embedding Generation Complete ===")
    print(f"Text embeddings saved to: {text_output_path}")
    print(f"Image embeddings saved to: {image_output_path}")

if __name__ == "__main__":
    main() 