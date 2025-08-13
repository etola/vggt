#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from utils.reconstruction_transform import (
    estimate_similarity_transform_from_recons,
    load_reconstruction,
    apply_similarity_transform_to_reconstruction,
    save_reconstruction_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align a source COLMAP reconstruction to a target using matched image names and camera centers."
    )
    parser.add_argument("--source", required=True, help="Path to source COLMAP sparse directory")
    parser.add_argument("--target", required=True, help="Path to target COLMAP sparse directory")
    parser.add_argument("--output", default=None, help="If set, directory to save transformed source reconstruction (text format)")
    parser.add_argument("--no_robust_scale", action="store_true", help="Disable robust scale estimation (use RMS ratio)")
    parser.add_argument("--json", dest="json_out", default=None, help="If set, write transform JSON to this path")
    return parser.parse_args()


def main():
    args = parse_args()

    result = estimate_similarity_transform_from_recons(
        source_sparse_dir=args.source,
        target_sparse_dir=args.target,
        robust_scale=(not args.no_robust_scale),
    )

    print("=== Similarity Transform (source -> target) ===")
    print(f"Common images: {result['num_common']}")
    if result['num_common'] <= 10:
        print(f"Names: {result['common_images']}")
    print(f"Scale: {result['scale']:.9f}")
    print("Rotation (3x3):")
    print(np.array2string(result['rotation'], formatter={'float_kind':lambda x: f"{x: .9f}"}))
    print(f"Translation: {np.array2string(result['translation'], formatter={'float_kind':lambda x: f'{x: .9f}'})}")
    print(f"RMSE (centers): {result['rmse']:.9f}")

    if args.json_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "scale": float(result['scale']),
                "rotation": result['rotation'].tolist(),
                "translation": result['translation'].tolist(),
                "rmse": float(result['rmse']),
                "num_common": int(result['num_common']),
                "common_images": result['common_images'],
            }, f, indent=2)
        print(f"Wrote transform JSON to {args.json_out}")

    if args.output is not None:
        source_rec = load_reconstruction(args.source)
        transformed = apply_similarity_transform_to_reconstruction(
            source_rec,
            scale=float(result['scale']),
            rotation=result['rotation'],
            translation=result['translation'],
            only_image_names=None,
        )
        save_reconstruction_text(transformed, args.output)
        print(f"Transformed source reconstruction saved to {args.output}")


if __name__ == "__main__":
    main() 