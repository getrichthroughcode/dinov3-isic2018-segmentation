from typing import List
import torch
import torchvision.utils as vutils


def OverlayMask(
    img: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
):
    overlay = vutils.draw_segmentation_masks(img, mask, alpha)
    return overlay.clamp(0, 1)


def MakeGrid(tensors: List[torch.Tensor], nrow: int = 4, pad: int = 2):
    tensors = [t.float() for t in tensors]
    return vutils.make_grid(tensors, nrow=nrow, padding=pad)


def SaveGrid(grid: torch.Tensor, path: str):
    vutils.save_image(grid.float(), path)
