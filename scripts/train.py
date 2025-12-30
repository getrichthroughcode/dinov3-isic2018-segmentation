# scripts/train.py

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dinoseg.training.trainer import Trainer, TrainerConfig
from dinoseg.models.baseline_unet import UNet, count_params
from dinoseg.models.dino_v3_unet import Dinov3UNet


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train segmentation models on ISIC2018 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "dinov3"],
        help="Model architecture to train",
    )
    model_group.add_argument(
        "--encoder",
        type=str,
        default="dinov3_vits16",
        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
        help="Encoder backbone for DINOv3 (only used if --model=dinov3)",
    )
    model_group.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights during training (only for DINOv3)",
    )
    model_group.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base channels for baseline UNet (only used if --model=baseline)",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data-root",
        type=str,
        default="data/isic2018",
        help="Path to ISIC2018 dataset root",
    )
    data_group.add_argument(
        "--image-size", type=int, default=256, help="Input image size (square)"
    )
    data_group.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use (0.0-1.0)",
    )

    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    train_group.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    train_group.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_group.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )

    # Regularization
    reg_group = parser.add_argument_group("Regularization")
    reg_group.add_argument(
        "--early-patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    reg_group.add_argument(
        "--early-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement for early stopping",
    )

    # System configuration
    sys_group = parser.add_argument_group("System Configuration")
    sys_group.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loading workers"
    )
    sys_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    sys_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    sys_group.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (auto-generated if not provided)",
    )
    output_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


def build_model(args):
    """Build model based on arguments."""
    if args.model == "baseline":
        print(f"Building Baseline UNet with {args.base_channels} base channels")
        model = UNet(n_channels=3, n_classes=1, base_ch=args.base_channels)

    elif args.model == "dinov3":
        freeze_str = "frozen" if args.freeze_encoder else "trainable"
        print(f"Building DINOv3 UNet with {args.encoder} encoder ({freeze_str})")
        model = Dinov3UNet(
            encoder_name=args.encoder, freeze_encoder=args.freeze_encoder
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Print parameter count
    num_params = count_params(model)
    print(f"Model parameters: {num_params:,}")

    return model


def build_config(args):
    """Build trainer configuration from arguments."""

    # Auto-generate output directory if not provided
    if args.output_dir is None:
        if args.model == "baseline":
            dir_name = f"baseline_unet_{int(args.data_fraction * 100)}percent"
        else:
            encoder_size = args.encoder.split("_")[
                1
            ]  # Extract 'vits16', 'vitb16', etc.
            freeze_str = "frozen" if args.freeze_encoder else "trainable"
            dir_name = f"dinov3_{encoder_size}_{freeze_str}_{int(args.data_fraction * 100)}percent"

        output_dir = f"runs/{dir_name}"
    else:
        output_dir = args.output_dir

    config = TrainerConfig(
        # Data
        root=args.data_root,
        size=args.image_size,
        fraction=args.data_fraction,
        # Training
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        # Regularization
        early_patience=args.early_patience,
        early_min_delta=args.early_min_delta,
        # System
        seed=args.seed,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        # Checkpointing
        output_dir=output_dir,
        resume_from=args.resume_from,
    )

    return config


def main():
    """Main training script."""
    args = parse_args()

    print("=" * 70)
    print("ISIC2018 Segmentation Training")
    print("=" * 70)

    # Build model and config
    model = build_model(args)
    config = build_config(args)

    # Create trainer
    trainer = Trainer(model, config)

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    best_checkpoint = trainer.train()

    print("=" * 70)
    print("Training completed successfully!")
    print(f"Best checkpoint saved at: {best_checkpoint}")
    print("=" * 70)


if __name__ == "__main__":
    main()
