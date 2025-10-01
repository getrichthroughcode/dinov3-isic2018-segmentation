from typing import List
import torch
import torchvision.utils as vutils


def OverlayMask(
    img: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
):
    overlay = vutils.draw_segmentation_masks(img, mask, alpha)
    print(type(overlay))
    return overlay


def MakeGrid(tensors: List[torch.Tensor], nrow: int = 4, pad: int = 2):
    tensors = [t.float() / 255.0 if t.dtype == torch.uint8 else t for t in tensors]
    return vutils.make_grid(tensors, nrow=nrow, padding=pad)


def SaveGrid(grid: torch.Tensor, path: str):
    vutils.save_image(grid.clamp(0, 1), path)
