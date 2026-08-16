import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.charbonnier import CharbonnierLoss

class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity (SSIM) Loss."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))
        
    def _create_window(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g[:, None] * g[None, :]
        return window_2d.unsqueeze(0).unsqueeze(0)
        
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.conv2d(pred, self.window, padding=self.window_size//2, groups=1)
        mu_target = F.conv2d(target, self.window, padding=self.window_size//2, groups=1)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=1) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=1) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=1) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
                   
        return 1 - ssim_map.mean()

class FFTL1Loss(nn.Module):
    """Frequency-domain L1 Loss."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        fft_pred = torch.fft.fft2(pred, norm="backward")
        fft_target = torch.fft.fft2(target, norm="backward")
        return F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))

class CompositeLoss(nn.Module):
    """Composite loss function combining Charbonnier, SSIM, and Frequency-domain L1."""
    def __init__(self, charbonnier_weight=1.0, ssim_weight=0.2, fft_weight=0.05, eps=1e-3):
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim = SSIMLoss()
        self.fft_l1 = FFTL1Loss()
        
    def forward(self, pred, target):
        loss = 0.0
        if self.charbonnier_weight > 0:
            loss += self.charbonnier_weight * self.charbonnier(pred, target)
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim(pred, target)
        if self.fft_weight > 0:
            loss += self.fft_weight * self.fft_l1(pred, target)
        return loss
