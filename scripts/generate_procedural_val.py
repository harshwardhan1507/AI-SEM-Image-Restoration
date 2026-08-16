import numpy as np
import cv2
from pathlib import Path
from verify_degradation import synthetic_degradation

def make_procedural(size=256):
    img = np.zeros((size, size), dtype=np.float32)
    # add some shapes
    num_shapes = np.random.randint(3, 10)
    for _ in range(num_shapes):
        shape_type = np.random.choice(['circle', 'rect', 'line'])
        color = np.random.uniform(0.3, 1.0)
        thickness = np.random.randint(1, 5)
        if shape_type == 'circle':
            center = (np.random.randint(0, size), np.random.randint(0, size))
            radius = np.random.randint(10, 50)
            cv2.circle(img, center, radius, color, thickness)
        elif shape_type == 'rect':
            pt1 = (np.random.randint(0, size), np.random.randint(0, size))
            pt2 = (np.random.randint(0, size), np.random.randint(0, size))
            cv2.rectangle(img, pt1, pt2, color, thickness)
        else:
            pt1 = (np.random.randint(0, size), np.random.randint(0, size))
            pt2 = (np.random.randint(0, size), np.random.randint(0, size))
            cv2.line(img, pt1, pt2, color, thickness)
            
    # Add base texture (flat)
    img = np.clip(img, 0, 1)
    return img

def main():
    np.random.seed(42)
    gt_dir = Path("dataset/val_ood/GT")
    noisy_dir = Path("dataset/val_ood/NoisyLR")
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(100):
        name = f"procedural_{i:04d}.npy"
        gt = make_procedural()
        # Synthetic degradation matches train params
        noisy, _, _ = synthetic_degradation(gt, sigma_min=0.0455, sigma_max=0.2405, poisson_prob=0.15)
        np.save(gt_dir / name, gt.astype(np.float32))
        np.save(noisy_dir / name, noisy.astype(np.float32))
        
    print(f"Generated 100 procedural OOD validation samples in {gt_dir.parent}")

if __name__ == "__main__":
    main()
