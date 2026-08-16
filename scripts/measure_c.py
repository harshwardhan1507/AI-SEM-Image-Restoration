import numpy as np
import cv2
import sys
import glob
from pathlib import Path
import scipy.optimize as opt

def measure_per_image(dataset_dir: str):
    gt_dir = Path(dataset_dir) / "GT"
    lr_dir = Path(dataset_dir) / "NoisyLR"
    
    gt_paths = sorted(glob.glob(str(gt_dir / "*.npy")))[:200]
    lr_paths = sorted(glob.glob(str(lr_dir / "*.npy")))[:200]
    
    if not gt_paths: return
    
    a_vals, b_vals, c_vals = [], [], []
    a_errs, b_errs, c_errs = [], [], []
    
    def model(mu, a, b, c):
        return a * mu**2 + b * mu + c
        
    for gt_p, lr_p in zip(gt_paths, lr_paths):
        hr_image = np.load(gt_p, mmap_mode="r").astype(np.float32)
        real_lr = np.load(lr_p, mmap_mode="r").astype(np.float32)
        
        hr_min, hr_max = hr_image.min(), hr_image.max()
        if hr_max > hr_min:
            hr_image = (hr_image - hr_min) / (hr_max - hr_min)
        else:
            hr_image = np.zeros_like(hr_image)
            
        h, w = hr_image.shape
        clean_lr = cv2.resize(hr_image, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        
        bins = np.linspace(0, 1, 60)
        inds = np.digitize(clean_lr, bins)
        
        mu_img = []
        var_img = []
        
        for i in range(1, len(bins)):
            mask = (inds == i)
            if np.sum(mask) > 10:
                mu_img.append(np.mean(clean_lr[mask]))
                var_img.append(np.var(real_lr[mask]))
                
        if len(mu_img) < 4:
            continue
            
        try:
            popt, pcov = opt.curve_fit(model, mu_img, var_img, p0=[0.02, 0, 0])
            perr = np.sqrt(np.diag(pcov))
            
            a_vals.append(popt[0])
            b_vals.append(popt[1])
            c_vals.append(popt[2])
            a_errs.append(perr[0])
            b_errs.append(perr[1])
            c_errs.append(perr[2])
        except Exception:
            continue
            
    a_median, b_median, c_median = np.median(a_vals), np.median(b_vals), np.median(c_vals)
    a_std, b_std, c_std = np.std(a_vals), np.std(b_vals), np.std(c_vals)
    
    print("=== Per-Image Fit Results ===")
    print(f"Median a (Gamma): {a_median:.6f} +/- {a_std:.6f}")
    print(f"Median b (Poisson): {b_median:.6f} +/- {b_std:.6f}")
    print(f"Median c (Gaussian): {c_median:.6f} +/- {c_std:.6f}")
    
    # Calculate how many b's and c's significantly differ from zero (i.e. abs(val) > 2*err)
    sig_b_mask = [np.abs(b) > 2*e for b, e in zip(b_vals, b_errs)]
    sig_c_mask = [np.abs(c) > 2*e for c, e in zip(c_vals, c_errs)]
    b_sig = sum(sig_b_mask)
    c_sig = sum(sig_c_mask)
    
    # 1. Sign census and correlation
    neg_b = sum(b < 0 for b, sig in zip(b_vals, sig_b_mask) if sig)
    print(f"Images with statistically significant Poisson (b): {b_sig} / {len(b_vals)}")
    print(f"  -> Of those significant, number with b < 0: {neg_b}")
    print(f"Images with statistically significant Additive (c): {c_sig} / {len(c_vals)}")
    
    corr = np.corrcoef(a_vals, b_vals)[0, 1]
    print(f"Correlation between 'a' and 'b': {corr:.4f}")

    # 2. The collinearity-free test
    print("\n=== Collinearity-Free Test (Shape) ===")
    # Accumulate std/mu over all pixels in the dataset, binned by mu
    mu_global = []
    std_global = []
    for gt_p, lr_p in zip(gt_paths, lr_paths):
        hr_image = np.load(gt_p, mmap_mode="r").astype(np.float32)
        real_lr = np.load(lr_p, mmap_mode="r").astype(np.float32)
        hr_min, hr_max = hr_image.min(), hr_image.max()
        if hr_max > hr_min: hr_image = (hr_image - hr_min) / (hr_max - hr_min)
        else: hr_image = np.zeros_like(hr_image)
        h, w = hr_image.shape
        clean_lr = cv2.resize(hr_image, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        
        bins = np.linspace(0.1, 0.9, 10)
        inds = np.digitize(clean_lr, bins)
        for i in range(1, len(bins)):
            mask = (inds == i)
            if np.sum(mask) > 100:
                mu_val = np.mean(clean_lr[mask])
                std_val = np.std(real_lr[mask])
                mu_global.append(mu_val)
                std_global.append(std_val / mu_val)  # std/mu
                
    mu_global = np.array(mu_global)
    std_global = np.array(std_global)
    
    # Average std_global over mu bins
    bins = np.linspace(0.1, 0.9, 10)
    inds = np.digitize(mu_global, bins)
    for i in range(1, len(bins)):
        mask = (inds == i)
        if np.sum(mask) > 0:
            print(f"mu ~ {bins[i-1]:.2f}: mean(std/mu) = {np.mean(std_global[mask]):.5f}")

if __name__ == "__main__":
    default_dir = (
        Path("dataset/train")
        if Path("dataset/train").exists()
        else (Path("datasets/train") if Path("datasets/train").exists() else Path("dataset/train/train"))
    )
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)
    measure_per_image(dataset_dir)
