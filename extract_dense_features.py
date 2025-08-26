#!/usr/bin/env python3
"""
Extract dense feature descriptors from VGGT DPT and Track heads.

This script demonstrates how to get dense feature maps from the VGGT model,
which can be useful for:
- Feature matching between images
- Dense correspondence estimation
- Visual similarity analysis
- Custom downstream tasks

Usage:
    python extract_dense_features.py --input_images /path/to/images --output_dir /path/to/features
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

# Import VGGT components
from vggt.models.vggt import VGGT
from vggt.heads.dpt_head import DPTHead
from demo_colmap import load_model, load_and_preprocess_images_square


def parse_args():
    parser = argparse.ArgumentParser(description="Extract dense feature descriptors from VGGT")
    parser.add_argument("--input_images", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("--model_name", type=str, default="facebook/vggt",
                        help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--resolution", type=int, default=518,
                        help="Input resolution for VGGT")
    parser.add_argument("--feature_type", type=str, choices=["dpt", "track", "both"], default="both",
                        help="Which features to extract")
    parser.add_argument("--save_format", type=str, choices=["npy", "pt"], default="npy",
                        help="Feature save format")
    
    return parser.parse_args()


def create_feature_extractors(model, feature_dim_dpt=256, feature_dim_track=128):
    """
    Create feature extractors from the VGGT model.
    
    Args:
        model: Loaded VGGT model
        feature_dim_dpt: Feature dimension for DPT features
        feature_dim_track: Feature dimension for track features
    
    Returns:
        dpt_feature_extractor: DPT-based feature extractor (full resolution)
        track_feature_extractor: Track-based feature extractor (half resolution)
    """
    embed_dim = model.aggregator.embed_dim * 2  # Usually 2048 for VGGT
    
    # Create DPT feature extractor (full resolution)
    dpt_feature_extractor = DPTHead(
        dim_in=embed_dim,
        patch_size=model.aggregator.patch_size,
        features=feature_dim_dpt,
        feature_only=True,      # Extract features only
        down_ratio=1,           # Full resolution
        pos_embed=True,         # Include positional embedding
    )
    
    # Create track feature extractor (half resolution, like in TrackHead)
    track_feature_extractor = DPTHead(
        dim_in=embed_dim,
        patch_size=model.aggregator.patch_size,
        features=feature_dim_track,
        feature_only=True,      # Extract features only
        down_ratio=2,           # Half resolution (more efficient)
        pos_embed=False,        # No positional embedding
    )
    
    return dpt_feature_extractor, track_feature_extractor


def extract_dense_features(model, images, feature_type="both", dtype=torch.float16):
    """
    Extract dense features from VGGT model.
    
    Args:
        model: VGGT model
        images: Input images [B, 3, H, W]
        feature_type: "dpt", "track", or "both"
        dtype: Data type for computation
    
    Returns:
        Dictionary containing extracted features
    """
    B, _, H, W = images.shape
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Get aggregated tokens from the backbone
            images_input = images.unsqueeze(0)  # [1, B, 3, H, W]
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_input)
    
    features = {}
    
    if feature_type in ["dpt", "both"]:
        # Extract DPT features (full resolution)
        embed_dim = model.aggregator.embed_dim * 2
        dpt_extractor = DPTHead(
            dim_in=embed_dim,
            patch_size=model.aggregator.patch_size,
            features=256,
            feature_only=True,
            down_ratio=1,
            pos_embed=True,
        ).to(images.device)
        
        dpt_features = dpt_extractor(aggregated_tokens_list, images_input, patch_start_idx)
        # Shape: [1, B, 256, H, W] -> [B, 256, H, W]
        features["dpt"] = dpt_features.squeeze(0).cpu().numpy()
    
    if feature_type in ["track", "both"]:
        # Extract track features (half resolution, like TrackHead)
        if hasattr(model, 'track_head') and model.track_head is not None:
            # Use the existing track head feature extractor
            track_features = model.track_head.feature_extractor(aggregated_tokens_list, images_input, patch_start_idx)
            # Shape: [1, B, 128, H//2, W//2] -> [B, 128, H//2, W//2]
            features["track"] = track_features.squeeze(0).cpu().numpy()
        else:
            # Create a track-style feature extractor
            embed_dim = model.aggregator.embed_dim * 2
            track_extractor = DPTHead(
                dim_in=embed_dim,
                patch_size=model.aggregator.patch_size,
                features=128,
                feature_only=True,
                down_ratio=2,
                pos_embed=False,
            ).to(images.device)
            
            track_features = track_extractor(aggregated_tokens_list, images_input, patch_start_idx)
            features["track"] = track_features.squeeze(0).cpu().numpy()
    
    return features


def save_features(features, image_names, output_dir, save_format="npy"):
    """Save extracted features to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    for feature_type, feature_maps in features.items():
        # feature_maps shape: [B, C, H, W]
        type_dir = os.path.join(output_dir, feature_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for i, image_name in enumerate(image_names):
            base_name = os.path.splitext(image_name)[0]
            
            if save_format == "npy":
                save_path = os.path.join(type_dir, f"{base_name}_features.npy")
                np.save(save_path, feature_maps[i])
            elif save_format == "pt":
                save_path = os.path.join(type_dir, f"{base_name}_features.pt")
                torch.save(torch.from_numpy(feature_maps[i]), save_path)
            
            print(f"  Saved {feature_type} features: {save_path}")


def load_images_from_directory(image_dir, max_images=None):
    """Load images from directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in valid_extensions:
        image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    image_paths = sorted(image_paths)
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    return [str(p) for p in image_paths]


def main():
    args = parse_args()
    
    print("🚀 Loading VGGT model...")
    model = load_model(args.model_name, enable_camera=False, enable_point=False, 
                      enable_depth=False, enable_track=True)  # Enable track for feature extraction
    model.eval()
    
    # Get image paths
    image_paths = load_images_from_directory(args.input_images)
    print(f"📁 Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    # Process images in batches
    all_features = {args.feature_type: [] for args.feature_type in ["dpt", "track", "both"]}
    if args.feature_type == "both":
        all_features = {"dpt": [], "track": []}
    else:
        all_features = {args.feature_type: []}
    
    processed_names = []
    
    for start_idx in range(0, len(image_paths), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        batch_names = [os.path.basename(p) for p in batch_paths]
        
        print(f"🔄 Processing batch {start_idx//args.batch_size + 1}: {len(batch_paths)} images")
        
        # Load and preprocess images
        images, _ = load_and_preprocess_images_square(batch_paths, args.resolution)
        images = images.cuda()
        
        # Extract features
        batch_features = extract_dense_features(model, images, args.feature_type)
        
        # Accumulate results
        for feature_type, features in batch_features.items():
            if feature_type not in all_features:
                all_features[feature_type] = []
            all_features[feature_type].append(features)
        
        processed_names.extend(batch_names)
        
        # Print feature info
        for feature_type, features in batch_features.items():
            print(f"  📊 {feature_type} features shape: {features.shape}")
    
    # Concatenate all batches
    print("\n🔗 Concatenating features from all batches...")
    for feature_type in all_features:
        if all_features[feature_type]:
            all_features[feature_type] = np.concatenate(all_features[feature_type], axis=0)
            print(f"  📊 Final {feature_type} features shape: {all_features[feature_type].shape}")
    
    # Save features
    print(f"\n💾 Saving features to {args.output_dir}...")
    save_features(all_features, processed_names, args.output_dir, args.save_format)
    
    # Create feature info file
    info_file = os.path.join(args.output_dir, "feature_info.txt")
    with open(info_file, "w") as f:
        f.write(f"VGGT Dense Feature Extraction Results\n")
        f.write(f"=====================================\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Input resolution: {args.resolution}\n")
        f.write(f"Feature type: {args.feature_type}\n")
        f.write(f"Save format: {args.save_format}\n")
        f.write(f"Processed images: {len(processed_names)}\n\n")
        
        for feature_type, features in all_features.items():
            if len(features) > 0:
                f.write(f"{feature_type.upper()} Features:\n")
                f.write(f"  Shape: {features.shape}\n")
                f.write(f"  Channels: {features.shape[1]}\n")
                f.write(f"  Spatial resolution: {features.shape[2]}x{features.shape[3]}\n")
                f.write(f"  File pattern: {feature_type}/{{}}_features.{args.save_format}\n\n")
    
    print(f"✅ Feature extraction complete!")
    print(f"📄 Feature info saved to: {info_file}")


if __name__ == "__main__":
    main() 