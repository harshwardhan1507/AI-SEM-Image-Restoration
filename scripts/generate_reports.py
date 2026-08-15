import os
import yaml
from pathlib import Path
import argparse

def generate_report(exp_id: str, loss_name: str):
    record_file = Path(f"outputs/experiments/{exp_id}_record.yaml")
    
    if not record_file.exists():
        print(f"Error: {record_file} not found. Did you run the training for {exp_id}?")
        return

    with open(record_file, "r") as f:
        record = yaml.safe_load(f)

    # Extract metrics
    metrics = record.get("metrics", {})
    val_psnr = metrics.get("val_psnr", {})
    val_ssim = metrics.get("val_ssim", {})
    val_lpips = metrics.get("val_lpips", {})

    best_psnr = val_psnr.get("best_value", "N/A")
    best_psnr_epoch = val_psnr.get("best_epoch", "N/A")
    
    best_ssim = val_ssim.get("best_value", "N/A")
    best_ssim_epoch = val_ssim.get("best_epoch", "N/A")
    
    best_lpips = val_lpips.get("best_value", "N/A")
    best_lpips_epoch = val_lpips.get("best_epoch", "N/A")

    if best_lpips == "N/A" or best_lpips is None:
        best_lpips_str = "Unavailable/Null (LPIPS likely failed to download/load pretrained weights)"
    else:
        best_lpips_str = str(best_lpips)

    report_content = f"""# {exp_id} Report ({loss_name} Loss)

## 1. Experiment Configuration
- **Dataset**: `./datasets` (Train/Val split)
- **Seed**: 42
- **Epochs**: 50
- **Batch Size**: 4
- **Optimizer**: AdamW (LR: 1e-3, Weight Decay: 1e-4, Min LR: 1e-6)
- **Scheduler**: CosineAnnealingLR
- **Loss**: {loss_name}
- **Architecture**: NAFNet (width=48, enc=[1,1,1], mid=1, dec=[1,1,1], ch=1, up=2)

## 2. PSNR Result
- **Best Validation PSNR**: {best_psnr}

## 3. SSIM Result
- **Best Validation SSIM**: {best_ssim}

## 4. LPIPS Result
- **Best Validation LPIPS**: {best_lpips_str}

## 5. Best Epoch for Each Metric
- **PSNR**: Epoch {best_psnr_epoch}
- **SSIM**: Epoch {best_ssim_epoch}
- **LPIPS**: Epoch {best_lpips_epoch}

## 6. Comparison against exp002 Charbonnier baseline
*(To be filled during final comparison analysis)*
- **exp002 Best PSNR**: ~29.9887 (Epoch 50)
- **exp002 Best SSIM**: ~0.8004 (Epoch 48)

## 7. Training Observations
*(To be filled based on Tensorboard or console logs)*
- Convergence speed: 
- Stability: 

## 8. Final Recommendation
*(To be filled after comparing all experiments)*

"""

    os.makedirs("experiments", exist_ok=True)
    report_path = f"experiments/{exp_id}_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"Generated {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate reports for exp004, exp005, exp006")
    args = parser.parse_args()

    if args.all:
        generate_report("exp004_nafnet_l1", "L1")
        generate_report("exp005_nafnet_mse", "MSE")
        generate_report("exp006_nafnet_psnr", "PSNR")
    else:
        print("Please run with --all")
