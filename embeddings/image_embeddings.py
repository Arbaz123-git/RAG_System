# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import glob

class ImagePatchDataset(Dataset):
    """Dataset for loading images and optionally dividing them into patches"""
    def __init__(self, image_dir, mask_dir=None, patch_size=224, transform=None, limit=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.transform = transform
        
        # Get all image paths
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, "*.png"))))
        
        # Limit the number of images if specified
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
            
        print(f"Found {len(self.image_paths)} images")
        
        # Pre-compute all patches to know the total dataset size
        self.patches_info = []
        self._precompute_patches()
        
    def _precompute_patches(self):
        """Pre-compute patch information for all images"""
        print("Pre-computing patch information...")
        for img_path in tqdm(self.image_paths):
            # Get corresponding mask path if available
            mask_path = None
            if self.mask_dir:
                img_name = os.path.basename(img_path)
                mask_path = os.path.join(self.mask_dir, img_name)
                
                # Check if mask exists
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {img_name}")
                    mask_path = None
            
            # Load image to get dimensions
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            
            if self.patch_size is None:
                # Use whole image
                self.patches_info.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'x': 0,
                    'y': 0,
                    'width': width,
                    'height': height
                })
            else:
                # Divide image into patches
                for y in range(0, height, self.patch_size):
                    for x in range(0, width, self.patch_size):
                        # Skip patches that would go out of bounds
                        if y + self.patch_size > height or x + self.patch_size > width:
                            continue
                            
                        self.patches_info.append({
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'x': x,
                            'y': y,
                            'width': self.patch_size,
                            'height': self.patch_size
                        })
        
        print(f"Total patches: {len(self.patches_info)}")
    
    def __len__(self):
        return len(self.patches_info)
    
    def __getitem__(self, idx):
        patch_info = self.patches_info[idx]
        
        # Load image
        img = Image.open(patch_info['img_path']).convert('RGB')
        
        # Extract patch
        if self.patch_size is not None:
            img = img.crop((
                patch_info['x'],
                patch_info['y'],
                patch_info['x'] + patch_info['width'],
                patch_info['y'] + patch_info['height']
            ))
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        # Load mask if available
        mask = None
        if patch_info['mask_path']:
            mask = Image.open(patch_info['mask_path']).convert('L')
            
            # Extract patch from mask
            if self.patch_size is not None:
                mask = mask.crop((
                    patch_info['x'],
                    patch_info['y'],
                    patch_info['x'] + patch_info['width'],
                    patch_info['y'] + patch_info['height']
                ))
                
            # Convert mask to tensor
            mask = transforms.ToTensor()(mask)
        
        return {
            'image': img,
            'mask': mask,
            'img_path': patch_info['img_path'],
            'patch_x': patch_info['x'],
            'patch_y': patch_info['y']
        }

def get_image_embeddings(model, dataloader, device):
    """Generate embeddings for images using the provided model"""
    embeddings = []
    metadata = []
    
    print("Generating image embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move images to device
            images = batch['image'].to(device)
            
            # Generate embeddings
            batch_embeddings = model.encode(images, convert_to_numpy=False)
            
            # Move embeddings to CPU and convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Add embeddings to list
            embeddings.append(batch_embeddings)
            
            # Add metadata
            for i in range(len(batch['img_path'])):
                metadata.append({
                    'img_path': batch['img_path'][i],
                    'patch_x': batch['patch_x'][i].item(),
                    'patch_y': batch['patch_y'][i].item()
                })
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    
    return embeddings, metadata

def main():
    parser = argparse.ArgumentParser(description='Generate image embeddings from Kvasir-SEG dataset')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory containing masks')
    parser.add_argument('--output_path', type=str, default='image_embeddings.npy', help='Path to save the embeddings')
    parser.add_argument('--model_name', type=str, default='clip-ViT-B-32', help='SentenceTransformers model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of patches (None for whole images)')
    parser.add_argument('--limit', type=int, default=1000, help='Limit the number of images to process')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda:0, etc.)')
    
    args = parser.parse_args()
    
    # Set patch_size to None if 0 is provided
    if args.patch_size == 0:
        args.patch_size = None
    
    # Determine device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=args.device)
    
    # Define image transformations
    if args.model_name.startswith('clip'):
        # CLIP models have their own preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        # Standard preprocessing for vision models
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    dataset = ImagePatchDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        patch_size=args.patch_size,
        transform=transform,
        limit=args.limit
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Generate embeddings
    embeddings, metadata = get_image_embeddings(model, dataloader, args.device)
    
    # Save embeddings
    print(f"Saving embeddings to {args.output_path}")
    np.save(args.output_path, embeddings)
    
    # Save metadata
    metadata_path = args.output_path.replace('.npy', '_metadata.npy')
    np.save(metadata_path, metadata)
    print(f"Metadata saved to {metadata_path}")
    
    print(f"Done! Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")

if __name__ == "__main__":
    main()

# python embeddings/image_embeddings.py --image_dir data/kvasir-seg/Kvasir-SEG/images --mask_dir data/kvasir-seg/Kvasir-SEG/masks --output_path output/image_embeddings.npy