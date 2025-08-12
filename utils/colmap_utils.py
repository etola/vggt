# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pycolmap
from typing import List, Dict, Tuple, Optional


def save_vggt_calibration_as_colmap(extrinsics_list: List[np.ndarray], 
                                   intrinsics_list: List[np.ndarray], 
                                   image_names_list: List[List[str]], 
                                   output_dir: str, 
                                   vggt_model_resolution: int = 518) -> str:
    """
    Save VGGT calibration in COLMAP format.
    
    Args:
        extrinsics_list: List of extrinsic matrices for each batch [B, 3, 4]
        intrinsics_list: List of intrinsic matrices for each batch [B, 3, 3]
        image_names_list: List of image names for each batch
        output_dir: Directory to save the COLMAP reconstruction
        vggt_model_resolution: Resolution used by VGGT model (default: 518)
    
    Returns:
        str: Path to the saved COLMAP reconstruction directory
    """
    reconstruction = pycolmap.Reconstruction()
    
    # Create cameras (using PINHOLE to preserve both fx and fy)
    camera_id = 1
    for batch_idx, intrinsics_batch in enumerate(intrinsics_list):
        for i, intrinsic in enumerate(intrinsics_batch):
            camera = pycolmap.Camera()
            camera.camera_id = camera_id
            camera.model = "PINHOLE"
            
            # Extract individual float values from the intrinsic matrix
            fx = float(intrinsic[0, 0])   # focal length x
            fy = float(intrinsic[1, 1])   # focal length y
            cx = float(intrinsic[0, 2])   # principal point x
            cy = float(intrinsic[1, 2])   # principal point y
            camera.params = [fx, fy, cx, cy]
            camera.width = vggt_model_resolution
            camera.height = vggt_model_resolution
            reconstruction.add_camera(camera)
            camera_id += 1
    
    # Create images with poses
    image_id = 1
    for batch_idx, (extrinsics, image_names) in enumerate(zip(extrinsics_list, image_names_list)):
        for i, (ext_matrix, img_name) in enumerate(zip(extrinsics, image_names)):
            # VGGT extrinsics are already in world-to-camera format
            R = ext_matrix[:3, :3]
            t = ext_matrix[:3, 3]
            
            # Create pycolmap Rigid3d object
            cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
            
            # Create image
            camera_id = batch_idx * len(extrinsics) + i + 1
            image = pycolmap.Image(
                id=image_id,
                name=img_name,
                camera_id=camera_id,
                cam_from_world=cam_from_world
            )
            
            # Mark as registered
            image.points2D = pycolmap.ListPoint2D([])
            image.registered = True
            
            reconstruction.add_image(image)
            image_id += 1
    
    # Save in ASCII format
    os.makedirs(output_dir, exist_ok=True)
    reconstruction.write_text(output_dir)
    print(f"VGGT calibration saved to {output_dir}")
    
    return output_dir


def load_colmap_calibration(colmap_dir: str) -> Dict:
    """
    Load COLMAP calibration from directory.
    
    Args:
        colmap_dir: Path to COLMAP sparse reconstruction directory
    
    Returns:
        dict: Dictionary containing loaded calibration data
    """
    try:
        reconstruction = pycolmap.Reconstruction(colmap_dir)
        
        # Extract camera parameters
        cameras = {}
        for camera_id, camera in reconstruction.cameras.items():
            cameras[camera_id] = {
                'model': camera.model,
                'width': camera.width,
                'height': camera.height,
                'params': camera.params
            }
        
        # Extract image poses
        images = {}
        for image_id, image in reconstruction.images.items():
            if image.registered:
                cam_from_world = image.cam_from_world
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                
                # Create 3x4 extrinsic matrix
                extrinsic = np.eye(3, 4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t
                
                # Create intrinsic matrix from camera parameters
                camera = reconstruction.cameras[image.camera_id]
                if camera.model.name == "SIMPLE_PINHOLE":
                    f, cx, cy = camera.params
                    intrinsic = np.array([
                        [f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]
                    ])
                elif camera.model.name == "PINHOLE":
                    fx, fy, cx, cy = camera.params
                    intrinsic = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])
                else:
                    # For other camera models, you might need to handle differently
                    intrinsic = np.eye(3)
                
                images[image.name] = {
                    'extrinsic': extrinsic,
                    'intrinsic': intrinsic,
                    'camera_id': image.camera_id,
                    'image_id': image_id
                }
        
        return {
            'cameras': cameras,
            'images': images,
            'reconstruction': reconstruction
        }
        
    except Exception as e:
        print(f"Error loading COLMAP reconstruction from {colmap_dir}: {e}")
        return None


def save_individual_camera_parameters(extrinsics: np.ndarray, 
                                    intrinsics: np.ndarray, 
                                    image_names: List[str], 
                                    output_dir: str) -> None:
    """
    Save individual camera parameters as numpy arrays.
    
    Args:
        extrinsics: Extrinsic matrices [B, 3, 4]
        intrinsics: Intrinsic matrices [B, 3, 3]
        image_names: List of image names
        output_dir: Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (extrinsic, intrinsic, image_name) in enumerate(zip(extrinsics, intrinsics, image_names)):
        base_name = os.path.splitext(image_name)[0]
        
        # Save extrinsic matrix
        extrinsic_file = os.path.join(output_dir, f"{base_name}_extrinsic.npy")
        np.save(extrinsic_file, extrinsic)
        
        # Save intrinsic matrix
        intrinsic_file = os.path.join(output_dir, f"{base_name}_intrinsic.npy")
        np.save(intrinsic_file, intrinsic)
    
    print(f"Saved {len(image_names)} camera parameter sets to {output_dir}")


def load_individual_camera_parameters(image_name: str, cameras_dir: str) -> Optional[Dict]:
    """
    Load individual camera parameters for a specific image.
    
    Args:
        image_name: Name of the image (with or without extension)
        cameras_dir: Directory containing camera parameter files
    
    Returns:
        dict: Dictionary with 'extrinsic' and 'intrinsic' keys, or None if not found
    """
    base_name = os.path.splitext(image_name)[0]
    
    extrinsic_file = os.path.join(cameras_dir, f"{base_name}_extrinsic.npy")
    intrinsic_file = os.path.join(cameras_dir, f"{base_name}_intrinsic.npy")
    
    if os.path.exists(extrinsic_file) and os.path.exists(intrinsic_file):
        extrinsic = np.load(extrinsic_file)
        intrinsic = np.load(intrinsic_file)
        
        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic
        }
    else:
        return None


def convert_vggt_to_colmap_format(extrinsics: np.ndarray, 
                                 intrinsics: np.ndarray, 
                                 image_names: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[str]]]:
    """
    Convert VGGT format camera parameters to COLMAP format.
    
    Args:
        extrinsics: VGGT extrinsic matrices [B, 3, 4]
        intrinsics: VGGT intrinsic matrices [B, 3, 3]
        image_names: List of image names
    
    Returns:
        tuple: (extrinsics_list, intrinsics_list, image_names_list) in COLMAP format
    """
    # For VGGT, the format is already compatible with COLMAP
    # Just need to wrap in lists for batch processing
    return [extrinsics], [intrinsics], [image_names]


def get_camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """
    Extract camera center from extrinsic matrix.
    
    Args:
        extrinsic: 3x4 extrinsic matrix [R|t]
    
    Returns:
        np.ndarray: Camera center in world coordinates [3,]
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # Camera center = -R^T * t
    camera_center = -R.T @ t
    return camera_center


def create_camera_info_dict(extrinsic: np.ndarray, 
                           intrinsic: np.ndarray, 
                           image_name: str) -> Dict:
    """
    Create a dictionary with camera information.
    
    Args:
        extrinsic: 3x4 extrinsic matrix
        intrinsic: 3x3 intrinsic matrix
        image_name: Name of the image
    
    Returns:
        dict: Dictionary with camera information
    """
    camera_center = get_camera_center_from_extrinsic(extrinsic)
    
    return {
        'image_name': image_name,
        'extrinsic': extrinsic,
        'intrinsic': intrinsic,
        'camera_center': camera_center,
        'rotation': extrinsic[:3, :3],
        'translation': extrinsic[:3, 3]
    } 