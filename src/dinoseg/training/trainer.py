# src/dinoseg/training/trainer.py

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as T
import any_gold as ag

from dinoseg.utils.seed import set_seed
from dinoseg.utils.metrics import SigmoidThreshold, DiceCoef, IoU


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None  # Fix: type annotation
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current validation metric (higher is better)

        Returns:
            True if should stop, False otherwise
        """
        if self.best is None:
            self.best = metric
            return False

        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False  # Fix: add explicit return
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    # Data
    root: str = "data/isic2018"
    size: int = 256
    fraction: float = 1.0

    # Training
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Regularization
    early_patience: int = 10
    early_min_delta: float = 0.0

    # System
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    # Checkpointing
    output_dir: str = "runs/experiment"
    resume_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class Trainer:
    """Generic trainer for segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        loss_fn: Optional[nn.Module] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration
            loss_fn: Loss function (default: BCEWithLogitsLoss)
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

        # Setup
        set_seed(config.seed)
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Build dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=config.mixed_precision and self.device.type == "cuda"
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_patience, min_delta=config.early_min_delta
        )

        # Tracking
        self.best_dice = -1.0
        self.current_epoch = 0

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _build_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader]:  # Fix: type annotation
        """Build train and validation dataloaders."""
        cfg = self.config
        transforms = T.Compose([T.Resize((cfg.size, cfg.size))])

        # Load datasets
        train_dataset = ag.ISIC2018SkinLesionDataset(
            root=cfg.root, split="train", transforms=transforms
        )
        val_dataset = ag.ISIC2018SkinLesionDataset(
            root=cfg.root, split="val", transforms=transforms
        )

        # Subsample if needed
        if cfg.fraction < 1.0:
            generator = torch.manual_seed(cfg.seed)

            # Subsample train
            num_train = int(len(train_dataset) * cfg.fraction)
            indices_train = torch.randperm(len(train_dataset), generator=generator)[
                :num_train
            ]
            train_dataset = Subset(train_dataset, indices_train.tolist())

            # Subsample val
            num_val = int(len(val_dataset) * cfg.fraction)
            indices_val = torch.randperm(len(val_dataset), generator=generator)[
                :num_val
            ]
            val_dataset = Subset(val_dataset, indices_val.tolist())

            print(
                f"Training on {cfg.fraction * 100:.1f}% of data: "
                f"{len(train_dataset)} train, {len(val_dataset)} val samples"
            )

        # Build dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _preprocess_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch data."""
        images = batch["image"].to(self.device, non_blocking=True).float()
        masks = batch["mask"].to(self.device, non_blocking=True).float()

        # Normalize masks to [0, 1]
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Fix mask dimensions
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        elif masks.dim() == 5 and masks.size(1) == 1:
            masks = masks.squeeze(1)

        return images, masks

    def _train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            images, masks = self._preprocess_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            if self.scaler.is_enabled():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        # Fix: explicit int conversion for len
        return total_loss / float(len(self.train_loader.dataset))  # type: ignore[arg-type]

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_samples = 0

        for batch in self.val_loader:
            images, masks = self._preprocess_batch(batch)

            # Forward pass
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)

            # Compute metrics
            predictions = SigmoidThreshold(logits)
            total_loss += loss.item() * images.size(0)
            total_dice += DiceCoef(predictions, masks).item()  # Fix: .item()
            total_iou += IoU(predictions, masks).item()  # Fix: .item()
            num_samples += images.size(0)

        avg_loss = total_loss / num_samples
        avg_dice = total_dice / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)

        return avg_loss, avg_dice, avg_iou

    def _save_checkpoint(self, filename: str, **extra_state: Any) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
            "config": self.config.to_dict(),
            **extra_state,
        }

        path = os.path.join(self.config.output_dir, filename)
        torch.save(state, path)

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def train(self) -> str:
        """
        Run full training loop.

        Returns:
            Path to best checkpoint
        """
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Output directory: {self.config.output_dir}")
        print(f"Device: {self.device}")
        print("-" * 70)

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()

            # Train & validate
            train_loss = self._train_one_epoch()
            val_loss, val_dice, val_iou = self._validate()

            # Update scheduler
            self.scheduler.step()

            # Log progress
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:03d}/{self.config.epochs}] "
                f"lr={lr:.2e} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # Save latest checkpoint
            self._save_checkpoint("latest.pt", dice=val_dice, iou=val_iou)

            # Save best checkpoint
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self._save_checkpoint("best.pt", dice=val_dice, iou=val_iou)
                print(f"  → New best model saved! Dice: {val_dice:.4f}")

            # Early stopping check
            if self.early_stopping.step(val_dice):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        best_path = os.path.join(self.config.output_dir, "best.pt")
        print("-" * 70)
        print(f"Training completed! Best Dice: {self.best_dice:.4f}")
        print(f"Best checkpoint: {best_path}")

        return best_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint for resuming training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_dice = checkpoint.get("best_dice", -1.0)

        print(
            f"Resumed from epoch {self.current_epoch}, best dice: {self.best_dice:.4f}"
        )
