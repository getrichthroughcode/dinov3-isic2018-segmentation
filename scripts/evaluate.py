import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import any_gold as ag
from pathlib import Path

from dinoseg.models.baseline_unet import UNet
from dinoseg.models.dino_v2_unet import DinoUNet
from dinoseg.models.dino_v3_unet import Dinov3UNet
from dinoseg.utils.metrics import DiceCoef, IoU, SigmoidThreshold


@torch.no_grad()
def evaluate(
    model_name, weights_path, input_size=256, batch_size=8, data_root="data/isic2018"
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} does not exist")

    ckpt = torch.load(weights_path, map_location=device)
    print(f"Loaded checkpoint from: {weights_path}")

    if model_name == "baseline":
        model = UNet(n_channels=3, n_classes=1, base_ch=32).to(device)
    elif model_name == "dinov2":
        model = DinoUNet(n_classes=1, encoder_name="dinov2_vits14").to(device)
    elif model_name == "dinov3s":
        model = Dinov3UNet(n_classes=1, encoder_name="dinov3_vits16").to(device)
    elif model_name == "dinov3b":
        model = Dinov3UNet(n_classes=1, encoder_name="dinov3_vitb16").to(device)
    elif model_name == "dinov3l":
        model = Dinov3UNet(n_classes=1, encoder_name="dinov3_vitl16").to(device)
    else:
        raise ValueError("Unknown model type. Choose among: baseline | dinov2 | dinov3")

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"Model loaded ({model_name})")

    transforms = T.Compose([T.Resize((input_size, input_size))])
    ds_test = ag.ISIC2018SkinLesionDataset(
        root=data_root, split="test", transforms=transforms
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    loss_fn = nn.BCEWithLogitsLoss()
    tot_loss, tot_dice, tot_iou, n = 0.0, 0.0, 0.0, 0

    for batch in dl_test:
        imgs = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()
        if masks.max() > 1.0:
            masks = masks / 255.0
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        logits = model(imgs)
        loss = loss_fn(logits, masks)
        preds = SigmoidThreshold(logits)

        tot_loss += loss.item() * imgs.size(0)
        tot_dice += DiceCoef(preds, masks)
        tot_iou += IoU(preds, masks)
        n += imgs.size(0)

    avg_loss = tot_loss / n
    avg_dice = (tot_dice / len(dl_test)).item()
    avg_iou = (tot_iou / len(dl_test)).item()

    print("Evaluation results (test split):")
    print(f"  Loss : {avg_loss:.4f}")
    print(f"  Dice : {avg_dice:.4f}")
    print(f"  IoU  : {avg_iou:.4f}")

    return avg_loss, avg_dice, avg_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "dinov2", "dinov3s", "dinov3b", "dinov3l"],
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/isic2018")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    evaluate(args.model, args.weights, args.size, args.batch, args.data_root)
