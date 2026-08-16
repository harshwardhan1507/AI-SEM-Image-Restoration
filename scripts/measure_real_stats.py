import numpy as np
import cv2
import sys
import glob
from pathlib import Path
from scipy.stats import skew
import os

def measure_real_stats(dataset_dir: str):
    gt_dir = Path(dataset_dir) / "GT"
    lr_dir = Path(dataset_dir) / "NoisyLR"
    
    gt_paths = sorted(glob.glob(str(gt_dir / "*.npy")))
    lr_paths = sorted(glob.glob(str(lr_dir / "*.npy")))
    
    if not gt_paths or not lr_paths:
        print(f"ERROR: Could not find GT or NoisyLR files in {dataset_dir}")
        sys.exit(1)
        
    # Limit to 200 samples for fast measurement
    gt_paths = gt_paths[:200]
    lr_paths = lr_paths[:200]
    
    print(f"Measuring stats on {len(gt_paths)} real KLA pairs...")
    
    global_min = float('inf')
    global_max = float('-inf')
    
    per_image_max_excess = []
    per_image_mins = []
    out_of_range_fractions = []
    excess_arrays = []
    
    relative_noise_stds = []
    dark_rel_noise_stds = []
    skews = []
    narrow_skews = []
    
    for gt_p, lr_p in zip(gt_paths, lr_paths):
        hr_image = np.load(gt_p, mmap_mode="r").astype(np.float32)
        real_lr = np.load(lr_p, mmap_mode="r").astype(np.float32)
        
        # Track global min/max and per-image min
        global_min = min(global_min, float(real_lr.min()))
        global_max = max(global_max, float(real_lr.max()))
        per_image_mins.append(float(real_lr.min()))
        
        # 1. Renormalize HR to [0, 1] to get the theoretical "clean_lr"
        hr_min = hr_image.min()
        hr_max = hr_image.max()
        if hr_max > hr_min:
            hr_image = (hr_image - hr_min) / (hr_max - hr_min)
        else:
            hr_image = np.zeros_like(hr_image)
            
        # 2. Downsample x2 with INTER_CUBIC (no anti-aliasing) to get expected clean_lr
        h, w = hr_image.shape
        lr_shape = (w // 2, h // 2)
        clean_lr = cv2.resize(hr_image, lr_shape, interpolation=cv2.INTER_CUBIC)
        
        # 3. Bright region mask stats
        mask = clean_lr > 0.8
        if np.sum(mask) > 10:
            # Noise is multiplicative: noise = real_lr / clean_lr
            noise_in_bright = real_lr[mask] / (clean_lr[mask] + 1e-8)  # Add epsilon just in case
            
            rel_noise_std = np.std(noise_in_bright)
            relative_noise_stds.append(rel_noise_std)
            
            noise_skew = skew(noise_in_bright)
            skews.append(noise_skew)
            
        # 3.5. Low-intensity (dark) statistics (mask < 0.3)
        dark_mask = clean_lr < 0.3
        if np.any(dark_mask):
            sig_dark = real_lr[dark_mask]
            mu_dark = clean_lr[dark_mask]
            if np.mean(mu_dark) > 1e-4:
                dark_rel_noise = np.std(sig_dark - mu_dark) / np.mean(mu_dark)
                dark_rel_noise_stds.append(dark_rel_noise)
                
        # 3.6. Narrow-band skew (0.08 < clean_lr < 0.12)
        narrow_mask = (clean_lr > 0.08) & (clean_lr < 0.12)
        if np.any(narrow_mask):
            sig_narrow = real_lr[narrow_mask]
            mu_narrow = clean_lr[narrow_mask]
            # Skew on residual with NO division
            residual_narrow = sig_narrow - mu_narrow
            if np.std(residual_narrow) > 1e-5:
                narrow_skews.append(skew(residual_narrow))
            
        # 4. Out of range fractions
        oor = (real_lr > 1.0) | (real_lr < 0.0)
        out_of_range_fractions.append(np.mean(oor))
        
        # 5. Excesses
        excess = real_lr[real_lr > 1.0] - 1.0
        if len(excess) > 0:
            per_image_max_excess.append(np.max(excess))
            excess_arrays.extend(excess)
        else:
            per_image_max_excess.append(0.0)

    # Aggregations
    median_oor = np.median(out_of_range_fractions)
    median_per_image_min = np.median(per_image_mins)
    
    if excess_arrays:
        median_per_image_max = np.median(per_image_max_excess)
        global_max_excess = np.max(per_image_max_excess)
        p999_pooled_excess = np.percentile(excess_arrays, 99.9)
    else:
        median_per_image_max = 0.0
        global_max_excess = 0.0
        p999_pooled_excess = 0.0
        
    avg_rel_noise_std = np.mean(relative_noise_stds)
    avg_dark_rel_noise = np.mean(dark_rel_noise_stds) if dark_rel_noise_stds else 0.0
    avg_skew = np.mean(skews)
    avg_narrow_skew = np.mean(narrow_skews) if narrow_skews else 0.0
    
    print("\n=== Real Data Baseline Statistics ===")
    print(f"Global Range: [{global_min:.5f}, {global_max:.5f}]")
    print(f"Median per-image min: {median_per_image_min:.5f}")
    print(f"Relative noise std (mask > 0.8): {avg_rel_noise_std:.5f}")
    print(f"Relative noise std (mask < 0.3): {avg_dark_rel_noise:.5f}")
    print(f"Noise Skew (mask > 0.8): {avg_skew:.5f}")
    print(f"Narrow-band Skew (0.08 < mu < 0.12): {avg_narrow_skew:.5f}")
    print(f"Median Out-of-range fraction: {median_oor*100:.3f}% (ratio: {median_oor:.5f})")
    print(f"Median of per-image max excess: {median_per_image_max:.5f}")
    print(f"Global max excess: {global_max_excess:.5f}")
    print(f"Pooled p99.9 excess: {p999_pooled_excess:.5f}")

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/krishna/cooks/AI-SEM-Image-Restoration/dataset/train"
    measure_real_stats(dataset_dir)
