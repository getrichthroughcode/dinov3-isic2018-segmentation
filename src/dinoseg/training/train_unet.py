import os
import time
import json
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from dinoseg.models.minimal_unet import UNet
from dinoseg.utils.metrics import SigmoidThreshold, DiceCoef, IoU
import any_gold as ag


@dataclass
class TrainCfg:
    root: str = "data/isic2018"
    size: int = 256
    batch: int = 8
    workers: int = 2
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    outdir: str = "runs/unet_baseline"
    seed: int = 0
    resume: Optional[str] = None
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def BuildLoaders(cfg: TrainCfg):
    tr = T.Compose([T.Resize((cfg.size, cfg.size))])
    ds_train = ag.ISIC2018SkinLesionDataset(root=cfg.root, split="train", transforms=tr)
    ds_val = ag.ISIC2018SkinLesionDataset(root=cfg.root, split="val", transforms=tr)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    return dl_train, dl_val


def SaveCheckpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def TrainOneEpoch(model, dl, optim, scaler, loss_fn, device):
    model.train()
    running = 0.0
    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True).float()
        masks = (
            batch["mask"].to(device, non_blocking=True).float().unsqueeze(1)
        )  # (B,1,H,W)

        optim.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optim.step()

        running += loss.item() * imgs.size(0)
    return running / len(dl.dataset)


@torch.no_grad()
def Eval(model, dl, loss_fn, device):
    model.eval()
    tot_loss, tot_dice, tot_iou, n = 0.0, 0.0, 0.0, 0
    for batch in dl:
        imgs = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float().unsqueeze(1)

        logits = model(imgs)
        loss = loss_fn(logits, masks)

        preds = SigmoidThreshold(logits)
        tot_loss += loss.item() * imgs.size(0)
        tot_dice += DiceCoef(preds, masks)
        tot_iou += IoU(preds, masks)
        n += imgs.size(0)
    return tot_loss / n, (tot_dice / len(dl)).item(), (tot_iou / len(dl)).item()


def TrainUNet(cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    dl_train, dl_val = BuildLoaders(cfg)

    model = UNet(n_channels=3, n_classes=1, base_ch=32).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.mixed_precision and device.type == "cuda"
    )

    best_dice = -1.0
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "cfg.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss = TrainOneEpoch(model, dl_train, optim, scaler, loss_fn, device)
        val_loss, val_dice, val_iou = Eval(model, dl_val, loss_fn, device)
        sched.step()

        lr = optim.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"lr={lr:.2e} train_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f} "
            f"({dt:.1f}s)"
        )

        SaveCheckpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "dice": val_dice,
            },
            os.path.join(cfg.outdir, "latest.pt"),
        )
        if val_dice > best_dice:
            best_dice = val_dice
            SaveCheckpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "dice": val_dice,
                },
                os.path.join(cfg.outdir, "best.pt"),
            )

    print(f"Best Dice: {best_dice:.4f} → {os.path.join(cfg.outdir, 'best.pt')}")

    return os.path.join(cfg.outdir, "best.pt")
