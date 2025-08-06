# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import gc

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import pycolmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Batch-Efficient Camera Calibration")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--resolution", type=int, default=256, help="Preprocessing resolution (higher = better quality, more memory). Model always runs at 518.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process together (higher = better poses, more memory)")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--cpu_offload", action="store_true", default=False, help="Offload model to CPU between batches")
    parser.add_argument("--no_rescale", action="store_true", default=False, help="Keep model resolution (518), don't rescale to original")
    return parser.parse_args()


def run_VGGT_batch(model, images, original_coords_batch, dtype, preprocessing_resolution, vggt_model_resolution=518):
    """
    Run VGGT for a batch of images to preserve multi-view geometry.
    
    Args:
        model: VGGT model with only camera head enabled
        images: [B, 3, H, W] batch of images (preprocessed at preprocessing_resolution)
        original_coords_batch: [B, 6] tensor with coordinate info for each image
        dtype: Data type for mixed precision
        preprocessing_resolution: Resolution used for preprocessing
        vggt_model_resolution: Fixed resolution for VGGT model (518)
    
    Returns:
        extrinsic: Camera extrinsic matrices [B, 3, 4]
        intrinsic: Camera intrinsic matrices [B, 3, 3]
        original_coords_batch: Coordinate tracking info
        scale_factor: Scale factor from preprocessing to model resolution
    """
    batch_size = images.shape[0]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Always resize to VGGT's trained resolution (518) before feeding to model
    scale_factor = vggt_model_resolution / preprocessing_resolution
    if images.shape[2] != vggt_model_resolution or images.shape[3] != vggt_model_resolution:
        images = F.interpolate(images, size=(vggt_model_resolution, vggt_model_resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Process as image sequence [B=1, S=batch_size, 3, H, W]
            images = images.unsqueeze(0)  # Add batch dimension -> [1, B, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict camera parameters for the batch
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Remove the outer batch dimension, keep the sequence dimension
    extrinsic = extrinsic.squeeze(0).cpu().numpy()  # [B, 3, 4]
    intrinsic = intrinsic.squeeze(0).cpu().numpy()  # [B, 3, 3]
    
    return extrinsic, intrinsic, original_coords_batch, scale_factor


def process_images_in_batches(model, image_paths, dtype, preprocessing_resolution, batch_size=8, cpu_offload=False, max_images=None, vggt_model_resolution=518):
    """
    Process images in batches to preserve multi-view geometry while managing memory.
    """
    all_extrinsics = []
    all_intrinsics = []
    all_original_coords = []
    all_scale_factors = []
    
    # Limit number of images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    # Split into batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    print(f"Processing {len(image_paths)} images in {num_batches} batches of size {batch_size}")
    print(f"Preprocessing at {preprocessing_resolution}x{preprocessing_resolution}, model runs at {vggt_model_resolution}x{vggt_model_resolution}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches}: images {start_idx + 1}-{end_idx}")
        
        # Load batch with proper aspect ratio preservation
        images, original_coords = load_and_preprocess_images_square(batch_paths, preprocessing_resolution)
        images = images.to(model.aggregator.camera_token.device)  # [B, 3, H, W]
        
        # Optionally move model to GPU for processing
        if cpu_offload:
            model = model.cuda()
        
        # Process batch with coordinate tracking
        extrinsic, intrinsic, coords, scale_factor = run_VGGT_batch(
            model, images, original_coords, dtype, preprocessing_resolution, vggt_model_resolution
        )
        
        # Collect results
        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_original_coords.append(coords.cpu().numpy())
        all_scale_factors.append(scale_factor)
        
        print(f"  Processed {len(batch_paths)} images in batch {batch_idx + 1}")
        
        # Aggressive memory cleanup
        del images
        del coords
        torch.cuda.empty_cache()
        gc.collect()
        
        # Optionally move model back to CPU
        if cpu_offload:
            model = model.cpu()
            torch.cuda.empty_cache()
    
    # Concatenate all results
    all_extrinsics = np.concatenate(all_extrinsics, axis=0)
    all_intrinsics = np.concatenate(all_intrinsics, axis=0) 
    all_original_coords = np.concatenate(all_original_coords, axis=0)
    
    return all_extrinsics, all_intrinsics, all_original_coords, all_scale_factors[0]  # All scale factors are the same


def extrinsic_3x4_to_4x4(ext_3x4):
    """Convert 3x4 extrinsic matrix [R|t] to 4x4 transformation matrix."""
    ext_4x4 = np.eye(4)
    ext_4x4[:3, :] = ext_3x4
    return ext_4x4


def rescale_camera_parameters_from_coords(reconstruction, original_coords, preprocessing_resolution, preprocessing_to_model_scale):
    """
    Rescale camera parameters from model resolution to original image resolutions.
    Uses coordinate tracking to properly handle aspect ratio preservation.
    """
    print("Rescaling camera parameters to original image dimensions...")
    
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        
        # Get coordinate info for this image (0-indexed)
        coords = original_coords[pyimageid - 1]
        x1, y1, x2, y2, orig_width, orig_height = coords
        
        # Calculate the scale factor from model resolution (518) to original max dimension
        max_orig_dim = max(orig_width, orig_height)
        preprocessing_to_original_scale = max_orig_dim / preprocessing_resolution
        model_to_original_scale = preprocessing_to_original_scale / preprocessing_to_model_scale
        
        # Scale camera parameters from model resolution to original
        pred_params = copy.deepcopy(pycamera.params)
        pred_params = pred_params * model_to_original_scale
        
        # Set principal point to center of original image 
        pred_params[-2:] = [orig_width / 2, orig_height / 2]
        
        # Update camera parameters
        pycamera.params = pred_params
        pycamera.width = int(orig_width)
        pycamera.height = int(orig_height)
        
        if pyimageid <= 3:  # Only print first few
            print(f"Image {pyimageid}: {int(orig_width)}x{int(orig_height)}, total_scale: {model_to_original_scale:.3f}")
    
    return reconstruction


def create_cameras_only_reconstruction(extrinsic, intrinsic, image_paths, image_size, shared_camera=False, camera_type="SIMPLE_PINHOLE"):
    """Create a COLMAP reconstruction with only camera parameters (no 3D points)."""
    reconstruction = pycolmap.Reconstruction()
    
    num_cameras = 1 if shared_camera else len(extrinsic)
    
    # Create cameras
    for cam_id in range(num_cameras):
        camera = pycolmap.Camera()
        camera.camera_id = cam_id + 1
        camera.model = camera_type
        
        if shared_camera:
            K = intrinsic[0]
        else:
            K = intrinsic[cam_id]
        
        if camera_type == "SIMPLE_PINHOLE":
            camera.params = [K[0, 0], K[0, 2], K[1, 2]]
        elif camera_type == "PINHOLE":
            camera.params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")
        
        camera.width = image_size[0]
        camera.height = image_size[1]
        reconstruction.add_camera(camera)
    
    # Create images (poses)
    for img_id, (ext_matrix, img_path) in enumerate(zip(extrinsic, image_paths)):
        debug_this_image = img_id < 3  # Only debug first 3 images
        
        if debug_this_image:
            print(f"Debug - Processing image {img_id + 1}: {os.path.basename(img_path)}")
            print(f"Debug - ext_matrix shape: {ext_matrix.shape}")
            print(f"Debug - ext_matrix:\n{ext_matrix}")
        
        # VGGT extrinsics are already in world-to-camera format (cam_from_world)
        # Extract rotation and translation directly 
        R = ext_matrix[:3, :3]
        t = ext_matrix[:3, 3]
        
        if debug_this_image:
            print(f"Debug - R shape: {R.shape}, t shape: {t.shape}")
            print(f"Debug - Translation magnitude: {np.linalg.norm(t):.6f}")
            print(f"Debug - Rotation angle (degrees): {np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi:.3f}")
        
        try:
            # Create pycolmap Rigid3d object for the pose
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(R), t
            )
            
            # Create image with proper constructor
            image = pycolmap.Image(
                id=img_id + 1,
                name=os.path.basename(img_path),
                camera_id=1 if shared_camera else img_id + 1,
                cam_from_world=cam_from_world
            )
            
            # Mark image as registered and add empty 2D points list
            image.points2D = pycolmap.ListPoint2D([])
            image.registered = True
            
            if debug_this_image:
                print(f"Debug - Successfully created Image for image {img_id + 1}")
            
            reconstruction.add_image(image)
            
        except Exception as e:
            print(f"Error creating pose for image {img_id + 1}: {e}")
            print(f"R determinant: {np.linalg.det(R)}")
            print(f"R is orthogonal: {np.allclose(R @ R.T, np.eye(3))}")
            raise
    
    return reconstruction


def demo_batch_efficient(args):
    """
    Batch-efficient camera calibration that processes images in small batches.
    Preserves multi-view geometry while managing memory usage.
    """
    print("Arguments:", vars(args))
    print("BATCH-EFFICIENT MODE: Processing images in batches to preserve multi-view geometry")
    print(f"Batch size: {args.batch_size} images per batch")
    print(f"Processing resolution: {args.resolution}x{args.resolution}")
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load model with only camera head enabled
    print("Loading VGGT model (camera-only, batch optimized)...")
    # VGGT is trained on 518x518 - we must keep this fixed
    vggt_model_resolution = 518
    model = VGGT(
        img_size=vggt_model_resolution,  # Fixed at 518 (model's trained resolution)
        enable_camera=True,
        enable_point=False,
        enable_depth=False,
        enable_track=False
    )
    
    # Enable gradient checkpointing for memory savings
    model.train()  # This enables checkpointing in the aggregator
    
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    
    # Filter state dict to only load camera-related parameters
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not any(skip_key in key for skip_key in ['depth_head', 'point_head', 'track_head']):
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    if not args.cpu_offload:
        model = model.to(device)
    else:
        print("CPU offloading enabled - model will be moved to GPU only during processing")
    
    print("Model loaded")

    # Get image paths
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    image_path_list = sorted(image_path_list)
    
    if args.max_images:
        print(f"Limiting to {args.max_images} images")
        image_path_list = image_path_list[:args.max_images]
    
    print(f"Found {len(image_path_list)} images")

    # Process images in batches to preserve multi-view geometry
    extrinsic, intrinsic, original_coords, preprocessing_to_model_scale = process_images_in_batches(
        model, image_path_list, dtype, args.resolution, 
        batch_size=args.batch_size, cpu_offload=args.cpu_offload, 
        max_images=args.max_images, vggt_model_resolution=vggt_model_resolution
    )
    
    print(f"Successfully processed {len(extrinsic)} images")

    # Create COLMAP reconstruction with only camera parameters (at model resolution)
    image_size = np.array([vggt_model_resolution, vggt_model_resolution])
    reconstruction = create_cameras_only_reconstruction(
        extrinsic, intrinsic, image_path_list, image_size,
        shared_camera=args.shared_camera, camera_type=args.camera_type
    )
    
    # Optionally rescale camera parameters to original image resolutions
    if not args.no_rescale:
        reconstruction = rescale_camera_parameters_from_coords(
            reconstruction, original_coords, args.resolution, preprocessing_to_model_scale
        )
    else:
        print(f"Keeping model resolution {vggt_model_resolution}x{vggt_model_resolution} (no rescaling)")

    # Debug final reconstruction state
    print(f"Debug - Final reconstruction summary:")
    print(f"  Number of cameras: {len(reconstruction.cameras)}")
    print(f"  Number of images: {len(reconstruction.images)}")
    print(f"  Number of 3D points: {len(reconstruction.points3D)}")

    # Save reconstruction in ASCII format
    output_dir = os.path.join(args.scene_dir, "sparse_batch_efficient")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        reconstruction.write_text(output_dir)
        print(f"Debug - Successfully wrote reconstruction to {output_dir}")
        
        # Check if files were actually created
        files = os.listdir(output_dir)
        print(f"Debug - Files created: {files}")
        
    except Exception as e:
        print(f"Error writing reconstruction: {e}")
        raise
    
    print(f"Camera calibration completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(extrinsic)} images with multi-view geometry preservation")
    print(f"Preprocessing resolution: {args.resolution}x{args.resolution}, Model resolution: {vggt_model_resolution}x{vggt_model_resolution}")
    if not args.no_rescale:
        print(f"Camera parameters rescaled to original image dimensions")
    else:
        print(f"Camera parameters kept at model resolution {vggt_model_resolution}x{vggt_model_resolution}")

    return True


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_batch_efficient(args) 