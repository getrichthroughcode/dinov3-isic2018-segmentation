import torch


def SigmoidThreshold(logits: torch.Tensor, thr: float = 0.5):
    return (logits.sigmoid() > thr).float()


def DiceCoef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    # pred, target: (B, 1, H, W) in {0,1}
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def IoU(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()
