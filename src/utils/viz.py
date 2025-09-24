from typing import List
import torch
import torchvision.utils as vutils
import torchvision.utils.draw_segmentation_masks as vdraw
import numpy as np
from PIL import Image


def Denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x * std + mean).clamp(0, 1)


def OverlayMask(
    img: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float = 0.5,
):
    #img_dn = Denorm(img, mean, std)
    overlay = vdraw(img, mask, alpha)
    return overlay.clamp(0, 1)


def MakeGrid(tensors: List[torch.Tensor], nrow: int = 4, pad: int = 2):
    return vutils.make_grid(tensors, nrow=nrow, padding=pad)


def SaveGrid(grid: torch.Tensor, path: str):
    arr = (grid.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr).save(path)
