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
    parser = argparse.ArgumentParser(description="VGGT Camera Calibration Only")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    return parser.parse_args()


def run_VGGT_calibration_only(model, images, dtype, resolution=518):
    """
    Run VGGT for camera calibration only (no depth/point cloud generation).
    This significantly reduces GPU memory usage.
    
    Args:
        model: VGGT model with only camera head enabled
        images: [B, 3, H, W] input images
        dtype: Data type for mixed precision
        resolution: Input resolution for VGGT
    
    Returns:
        extrinsic: Camera extrinsic matrices
        intrinsic: Camera intrinsic matrices
    """
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras ONLY (no depth prediction)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic


def create_cameras_only_reconstruction(extrinsic, intrinsic, image_paths, image_size, shared_camera=False, camera_type="SIMPLE_PINHOLE"):
    """
    Create a COLMAP reconstruction with only camera parameters (no 3D points).
    """
    reconstruction = pycolmap.Reconstruction()
    
    num_cameras = 1 if shared_camera else len(extrinsic)
    
    # Create cameras
    for cam_id in range(num_cameras):
        camera = pycolmap.Camera()
        camera.camera_id = cam_id + 1
        camera.model = camera_type
        
        if shared_camera:
            # Use the first camera's intrinsic parameters for all
            K = intrinsic[0]
        else:
            K = intrinsic[cam_id]
        
        if camera_type == "SIMPLE_PINHOLE":
            # SIMPLE_PINHOLE: f, cx, cy
            camera.params = [K[0, 0], K[0, 2], K[1, 2]]
        elif camera_type == "PINHOLE":
            # PINHOLE: fx, fy, cx, cy
            camera.params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")
        
        camera.width = image_size[0]
        camera.height = image_size[1]
        reconstruction.add_camera(camera)
    
    # Create images (poses)
    for img_id, (ext_matrix, img_path) in enumerate(zip(extrinsic, image_paths)):
        image = pycolmap.Image()
        image.image_id = img_id + 1
        image.camera_id = 1 if shared_camera else img_id + 1
        image.name = os.path.basename(img_path)
        
        # Convert from camera-to-world to world-to-camera (COLMAP convention)
        world_to_camera = np.linalg.inv(ext_matrix)
        R = world_to_camera[:3, :3]
        t = world_to_camera[:3, 3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        quat = rotation_matrix_to_quaternion(R)
        image.qvec = quat
        image.tvec = t
        
        reconstruction.add_image(image)
    
    return reconstruction


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])


def rescale_camera_parameters(reconstruction, original_coords, img_load_resolution, vggt_resolution):
    """Rescale camera parameters to match original image resolutions."""
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        
        # Get original image dimensions
        real_image_size = original_coords[pyimageid - 1, -2:]
        resize_ratio = max(real_image_size) / vggt_resolution
        
        # Scale camera parameters
        pred_params = copy.deepcopy(pycamera.params)
        pred_params = pred_params * resize_ratio
        
        # Set principal point to image center
        real_pp = real_image_size / 2
        if len(pred_params) == 3:  # SIMPLE_PINHOLE
            pred_params[-2:] = real_pp
        elif len(pred_params) == 4:  # PINHOLE
            pred_params[-2:] = real_pp
        
        pycamera.params = pred_params
        pycamera.width = int(real_image_size[0])
        pycamera.height = int(real_image_size[1])
    
    return reconstruction


def demo_calibration_only(args):
    """
    Demo function that only performs camera calibration without point cloud generation.
    This uses significantly less GPU memory than the full reconstruction pipeline.
    """
    # Print configuration
    print("Arguments:", vars(args))
    print("CALIBRATION-ONLY MODE: No depth maps or point clouds will be generated")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load VGGT model with ONLY camera head enabled (saves GPU memory)
    print("Loading VGGT model with only camera calibration enabled...")
    model = VGGT(
        enable_camera=True,   # Keep camera head for pose estimation
        enable_point=False,   # Disable point head to save memory
        enable_depth=False,   # Disable depth head to save memory  
        enable_track=False    # Disable track head to save memory
    )
    
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    
    # Filter state dict to only load camera-related parameters
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not any(skip_key in key for skip_key in ['depth_head', 'point_head', 'track_head']):
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    print(f"Model loaded (calibration-only mode)")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Sort for consistent ordering
    image_path_list = sorted(image_path_list)
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT for camera calibration ONLY (no depth/point cloud)
    print("Running camera calibration...")
    extrinsic, intrinsic = run_VGGT_calibration_only(model, images, dtype, vggt_fixed_resolution)
    
    print(f"Estimated camera parameters for {len(extrinsic)} images")
    print(f"Intrinsic matrix shape: {intrinsic.shape}")
    print(f"Extrinsic matrix shape: {extrinsic.shape}")

    # Create COLMAP reconstruction with only camera parameters
    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    reconstruction = create_cameras_only_reconstruction(
        extrinsic, intrinsic, image_path_list, image_size, 
        shared_camera=args.shared_camera, camera_type=args.camera_type
    )

    # Rescale camera parameters to original image resolutions
    reconstruction = rescale_camera_parameters(
        reconstruction, original_coords.cpu().numpy(), 
        img_load_resolution, vggt_fixed_resolution
    )

    # Rename images to original names
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pyimage.name = base_image_path_list[pyimageid - 1]

    # Save reconstruction (cameras and poses only, no 3D points)
    print(f"Saving camera calibration to {args.scene_dir}/sparse_calibration_only")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse_calibration_only")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    
    print("Camera calibration completed successfully!")
    print(f"Results saved to: {sparse_reconstruction_dir}")
    print("Note: No 3D points or point cloud generated (calibration-only mode)")

    return True


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_calibration_only(args) 