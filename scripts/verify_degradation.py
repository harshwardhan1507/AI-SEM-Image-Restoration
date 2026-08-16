import numpy as np
import cv2
import sys
import glob
from pathlib import Path
from scipy.stats import skew

def synthetic_degradation(hr_image: np.ndarray, sigma_min: float, sigma_max: float, poisson_prob: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    hr_min = hr_image.min()
    hr_max = hr_image.max()
    if hr_max > hr_min:
        hr_image = (hr_image - hr_min) / (hr_max - hr_min)
    else:
        hr_image = np.zeros_like(hr_image)
        
    h, w = hr_image.shape
    lr_shape = (w // 2, h // 2)
    lr_image = cv2.resize(hr_image, lr_shape, interpolation=cv2.INTER_CUBIC)
    
    sigma = np.random.uniform(sigma_min, sigma_max)
    L = 1.0 / (sigma ** 2)
    
    # Gamma (Multiplicative)
    noise = np.random.gamma(shape=L, scale=1.0/L, size=lr_image.shape).astype(np.float32)
    synthetic_lr = lr_image * noise
    
    # Poisson (Shot Noise)
    if poisson_prob > 0.0:
        scale = np.random.uniform(110.0, 180.0)
        clean_pos = np.clip(synthetic_lr, 0, None)
        clean_neg = np.clip(synthetic_lr, None, 0)
        noisy_pos = np.random.poisson(clean_pos * scale) / scale
        synthetic_lr = (noisy_pos + clean_neg).astype(np.float32)
    
    return synthetic_lr, lr_image, sigma


def run_verification(dataset_dir: str, poisson_prob: float, sigma_min: float, sigma_max: float):
    np.random.seed(42)
    
    gt_paths = sorted(glob.glob(str(Path(dataset_dir) / "**" / "*.npy"), recursive=True))[:200]
        
    if not gt_paths:
        print(f"ERROR: No .npy files found in {dataset_dir}")
        sys.exit(1)
        
    print(f"\n=== Verification: sigma ~ U({sigma_min}, {sigma_max}), Poisson={poisson_prob} ===")
    
    gt_images = [np.load(p, mmap_mode="r").astype(np.float32) for p in gt_paths]
    
    mean_ratios = []
    relative_noise_stds = []
    dark_rel_noise_stds = []
    skews = []
    narrow_skews = []
    out_of_range_fractions = []
    per_image_max_excess = []
    per_image_mins = []
    
    global_min = float('inf')
    
    for hr_image in gt_images:
        synthetic_lr, clean_lr, sigma = synthetic_degradation(hr_image, sigma_min, sigma_max, poisson_prob)
        
        global_min = min(global_min, float(synthetic_lr.min()))
        per_image_mins.append(float(synthetic_lr.min()))
        
        mean_ratio = synthetic_lr.mean() / (clean_lr.mean() + 1e-8)
        mean_ratios.append(mean_ratio)
        
        # 1. High-intensity statistics (mask > 0.8)
        mask = clean_lr > 0.8
        if np.sum(mask) > 10:
            noise_in_bright = synthetic_lr[mask] / (clean_lr[mask] + 1e-8)
            relative_noise_stds.append(np.std(noise_in_bright))
            skews.append(skew(noise_in_bright))
            
        # 2. Low-intensity (dark) statistics (mask < 0.3)
        dark_mask = clean_lr < 0.3
        if np.sum(dark_mask) > 10:
            sig_dark = synthetic_lr[dark_mask]
            mu_dark = clean_lr[dark_mask]
            dark_rel_noise = np.std(sig_dark - mu_dark) / np.mean(mu_dark)
            dark_rel_noise_stds.append(dark_rel_noise)
            
        # 3. Narrow-band skew (0.08 < clean_lr < 0.12)
        narrow_mask = (clean_lr > 0.08) & (clean_lr < 0.12)
        if np.any(narrow_mask):
            sig_narrow = synthetic_lr[narrow_mask]
            mu_narrow = clean_lr[narrow_mask]
            residual_narrow = sig_narrow - mu_narrow
            if np.std(residual_narrow) > 1e-5:
                narrow_skews.append(skew(residual_narrow))
        
        oor = (synthetic_lr > 1.0) | (synthetic_lr < 0.0)
        out_of_range_fractions.append(np.mean(oor))
        
        excess = synthetic_lr[synthetic_lr > 1.0] - 1.0
        if len(excess) > 0:
            per_image_max_excess.append(np.max(excess))
        else:
            per_image_max_excess.append(0.0)
            
    avg_mean_ratio = np.mean(mean_ratios)
    avg_rel_noise_std = np.mean(relative_noise_stds)
    avg_dark_rel_noise_std = np.mean(dark_rel_noise_stds) if dark_rel_noise_stds else 0.0
    avg_skew = np.mean(skews)
    avg_narrow_skew = np.mean(narrow_skews) if narrow_skews else 0.0
    median_oor_fraction = np.median(out_of_range_fractions)
    median_per_image_max = np.median(per_image_max_excess)
    median_per_image_min = np.median(per_image_mins)
    
    print("\n--- Calibration Results ---")
    print(f"Global min: {global_min:.5f} (diagnostic)")
    print(f"Median per-image min: {median_per_image_min:.5f} (target: ~ -0.002)")
    print(f"Mean preservation ratio: {avg_mean_ratio:.5f}")
    print(f"Relative noise std (mask > 0.8): {avg_rel_noise_std:.5f}")
    print(f"Relative noise std (mask < 0.3): {avg_dark_rel_noise_std:.5f}")
    print(f"Noise Skew (mask > 0.8): {avg_skew:.5f}")
    print(f"Narrow-band Skew (0.08 < mu < 0.12): {avg_narrow_skew:.5f}")
    print(f"Median Out-of-range pixel fraction: {median_oor_fraction*100:.3f}% (ratio: {median_oor_fraction:.5f})")
    print(f"Median of per-image max excess: {median_per_image_max:.5f}")

    checks = [
        # (name,                 measured,             target,  tolerance_rel)
        ("mean_ratio",           avg_mean_ratio,       1.0001,  0.001), 
        ("rel_noise_std",        avg_rel_noise_std,    0.1657,  0.15),
        ("dark_rel_noise",       avg_dark_rel_noise_std,0.2486, 0.15),
        ("noise_skew",           avg_skew,             0.2960,  0.15),
        ("oor_fraction",         median_oor_fraction,  0.00702, 0.30),
        ("median_per_image_max", median_per_image_max, 0.359,   0.25),
    ]
    
    failed_tolerance = []
    signs = []
    
    print("\n--- Error Analysis ---")
    for name, val, target, tol in checks:
        if name == "mean_ratio":
            continue
        err_rel = (val - target) / target
        sign = 1 if err_rel > 0 else -1
        signs.append(sign)
        status = "FAIL" if abs(err_rel) > tol else "PASS"
        if status == "FAIL":
            failed_tolerance.append(name)
        print(f"{name:20s}: err {err_rel*100:6.2f}% ({status})")
        
    if len(set(signs)) == 1:
        print("\nSYSTEMATIC BIAS DETECTED: All checks share the same error sign.")
        return False
        
    if failed_tolerance:
        print(f"\nFAILED TOLERANCE: {failed_tolerance}")
        return False

    print("\nVERIFICATION PASSED! Straddled zero and within tolerances.")
    return True

if __name__ == "__main__":
    dataset_path = "/Users/krishna/cooks/AI-SEM-Image-Restoration/dataset/train/GT"
    
    print("=======================================")
    print("Sweeping sigma_max for EXP009")
    print("=======================================")
    
    sigma_min = 0.040
    for sigma_max in [0.213, 0.222, 0.231, 0.2405]:
        passed = run_verification(dataset_path, poisson_prob=1.0, sigma_min=sigma_min, sigma_max=sigma_max)
        if passed:
            print(f">>> Found optimal sigma_max = {sigma_max}")
