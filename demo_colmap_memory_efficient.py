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
    parser = argparse.ArgumentParser(description="VGGT Memory-Efficient Camera Calibration")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--resolution", type=int, default=256, help="Preprocessing resolution (higher = better quality, more memory). Model always runs at 518.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--cpu_offload", action="store_true", default=False, help="Offload model to CPU between images")
    parser.add_argument("--no_rescale", action="store_true", default=False, help="Keep model resolution (518), don't rescale to original")
    return parser.parse_args()


def run_VGGT_single_image(model, image, original_coords, dtype, preprocessing_resolution, vggt_model_resolution=518):
    """
    Run VGGT for a single image only (most memory efficient).
    Properly handles aspect ratio preservation using the coordinate tracking.
    
    Args:
        model: VGGT model with only camera head enabled
        image: [3, H, W] single image tensor (preprocessed at preprocessing_resolution)
        original_coords: [6] tensor with [x1, y1, x2, y2, width, height] coordinate info
        dtype: Data type for mixed precision
        preprocessing_resolution: Resolution used for preprocessing
        vggt_model_resolution: Fixed resolution for VGGT model (518)
    
    Returns:
        extrinsic: Camera extrinsic matrix for this image
        intrinsic: Camera intrinsic matrix for this image  
        original_coords: Coordinate tracking info for this image
        scale_factor: Scale factor from preprocessing to model resolution
    """
    assert len(image.shape) == 3
    assert image.shape[0] == 3

    # Always resize to VGGT's trained resolution (518) before feeding to model
    scale_factor = vggt_model_resolution / preprocessing_resolution
    if image.shape[1] != vggt_model_resolution or image.shape[2] != vggt_model_resolution:
        image = image.unsqueeze(0)  # [1, 3, H, W]
        image = F.interpolate(image, size=(vggt_model_resolution, vggt_model_resolution), mode="bilinear", align_corners=False)
        image = image.squeeze(0)  # Back to [3, H, W]

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Process as single image sequence [B=1, S=1, 3, H, W]
            image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(image)

        # Predict camera parameters for single image
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image.shape[-2:])

    # Remove batch and sequence dimensions
    extrinsic = extrinsic.squeeze(0).squeeze(0).cpu().numpy()  # [3, 4]
    intrinsic = intrinsic.squeeze(0).squeeze(0).cpu().numpy()  # [3, 3]
    
    return extrinsic, intrinsic, original_coords, scale_factor


def process_images_sequentially(model, image_paths, dtype, preprocessing_resolution, cpu_offload=False, max_images=None, vggt_model_resolution=518):
    """
    Process images one by one to minimize memory usage.
    Properly preserves aspect ratio using coordinate tracking.
    """
    extrinsics = []
    intrinsics = []
    all_original_coords = []
    all_scale_factors = []
    
    # Limit number of images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    print(f"Processing {len(image_paths)} images sequentially...")
    print(f"Preprocessing at {preprocessing_resolution}x{preprocessing_resolution}, model runs at {vggt_model_resolution}x{vggt_model_resolution}")
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Load single image with proper aspect ratio preservation
        images, original_coords = load_and_preprocess_images_square([img_path], preprocessing_resolution)
        image = images[0].to(model.aggregator.camera_token.device)  # [3, H, W]
        coords = original_coords[0]  # [6] - coordinate info for this image
        
        # Optionally move model to GPU for processing
        if cpu_offload:
            model = model.cuda()
        
        # Process single image with coordinate tracking
        extrinsic, intrinsic, coords, scale_factor = run_VGGT_single_image(
            model, image, coords, dtype, preprocessing_resolution, vggt_model_resolution
        )
        
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)
        all_original_coords.append(coords.cpu().numpy())
        all_scale_factors.append(scale_factor)
        
        # Aggressive memory cleanup
        del image
        del images
        del coords
        torch.cuda.empty_cache()
        gc.collect()
        
        # Optionally move model back to CPU
        if cpu_offload:
            model = model.cpu()
            torch.cuda.empty_cache()
    
    return np.stack(extrinsics), np.stack(intrinsics), np.stack(all_original_coords), all_scale_factors[0]  # All scale factors are the same


def extrinsic_3x4_to_4x4(ext_3x4):
    """Convert 3x4 extrinsic matrix [R|t] to 4x4 transformation matrix."""
    ext_4x4 = np.eye(4)
    ext_4x4[:3, :] = ext_3x4
    return ext_4x4


def rescale_camera_parameters_from_coords(reconstruction, original_coords, preprocessing_resolution, preprocessing_to_model_scale):
    """
    Rescale camera parameters from model resolution to original image resolutions.
    Uses coordinate tracking to properly handle aspect ratio preservation.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
        original_coords: Array of shape (N, 6) with [x1, y1, x2, y2, width, height] for each image
        preprocessing_resolution: The resolution used during preprocessing (e.g., 256, 1000)
        preprocessing_to_model_scale: Scale factor from preprocessing to model (518/preprocessing_resolution)
    """
    print("Rescaling camera parameters to original image dimensions...")
    
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        
        # Get coordinate info for this image (0-indexed)
        coords = original_coords[pyimageid - 1]
        x1, y1, x2, y2, orig_width, orig_height = coords
        
        # Calculate the scale factor from model resolution (518) to original max dimension
        # Chain: original -> preprocessing -> model (518)
        # So: model -> original = (original/preprocessing) * (preprocessing/model)
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
            print(f"Debug - R:\n{R}")
            print(f"Debug - t: {t}")
        
        try:
            # Create pycolmap Rigid3d object for the pose
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(R), t
            )
            if debug_this_image:
                print(f"Debug - Successfully created Rigid3d for image {img_id + 1}")
            
            # Create image with proper constructor
            image = pycolmap.Image(
                id=img_id + 1,
                name=os.path.basename(img_path),
                camera_id=1 if shared_camera else img_id + 1,
                cam_from_world=cam_from_world
            )
            
            # Mark image as registered and add empty 2D points list
            # This is required for pycolmap to write the image to output files
            image.points2D = pycolmap.ListPoint2D([])
            image.registered = True
            
            if debug_this_image:
                print(f"Debug - Successfully created Image for image {img_id + 1}")
                print(f"Debug - Image registered: {image.registered}")
                print(f"Debug - Image points2D length: {len(image.points2D)}")
            
            reconstruction.add_image(image)
            if debug_this_image:
                print(f"Debug - Successfully added image {img_id + 1} to reconstruction")
            
        except Exception as e:
            print(f"Error creating pose for image {img_id + 1}: {e}")
            print(f"R determinant: {np.linalg.det(R)}")
            print(f"R is orthogonal: {np.allclose(R @ R.T, np.eye(3))}")
            raise
    
    return reconstruction





def demo_memory_efficient(args):
    """
    Extremely memory-efficient camera calibration that processes images one by one.
    """
    print("Arguments:", vars(args))
    print("MEMORY-EFFICIENT MODE: Processing images individually")
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
    print("Loading VGGT model (camera-only, memory optimized)...")
    # VGGT is trained on 518x518 - we must keep this fixed
    vggt_model_resolution = 518
    model = VGGT(
        img_size=vggt_model_resolution,  # Fixed at 518 (model's trained resolution)
        enable_camera=True,
        enable_point=False,
        enable_depth=False,
        enable_track=False
    )
    
    # Enable gradient checkpointing for even more memory savings
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

    # Process images sequentially to minimize memory usage
    extrinsic, intrinsic, original_coords, preprocessing_to_model_scale = process_images_sequentially(
        model, image_path_list, dtype, args.resolution, 
        cpu_offload=args.cpu_offload, max_images=args.max_images, vggt_model_resolution=vggt_model_resolution
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
    
    if len(reconstruction.images) > 0:
        print(f"  First few image names:")
        for i, (img_id, img) in enumerate(reconstruction.images.items()):
            print(f"    Image {img_id}: {img.name}")
            if i >= 2:  # Show first 3
                break
    
    # Save reconstruction in ASCII format
    output_dir = os.path.join(args.scene_dir, "sparse_memory_efficient")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        reconstruction.write_text(output_dir)
        print(f"Debug - Successfully wrote reconstruction to {output_dir}")
        
        # Check if files were actually created
        files = os.listdir(output_dir)
        print(f"Debug - Files created: {files}")
        
        # Check if images.txt has content
        images_file = os.path.join(output_dir, "images.txt")
        if os.path.exists(images_file):
            with open(images_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f"Debug - images.txt has {len(lines)} lines")
                if len(lines) > 0:
                    print(f"Debug - First few lines of images.txt:")
                    for i, line in enumerate(lines[:5]):
                        print(f"  {i+1}: {line}")
        
    except Exception as e:
        print(f"Error writing reconstruction: {e}")
        raise
    
    print(f"Camera calibration completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(extrinsic)} images with aspect-ratio preservation")
    print(f"Preprocessing resolution: {args.resolution}x{args.resolution}, Model resolution: {vggt_model_resolution}x{vggt_model_resolution}")
    if not args.no_rescale:
        print(f"Camera parameters rescaled to original image dimensions")
    else:
        print(f"Camera parameters kept at model resolution {vggt_model_resolution}x{vggt_model_resolution}")

    return True


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_memory_efficient(args) 