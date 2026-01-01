# scripts/evaluate.py

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import any_gold as ag

from dinoseg.models.baseline_unet import UNet
from dinoseg.models.dino_v3_unet import Dinov3UNet
from dinoseg.utils.metrics import DiceCoef, IoU, SigmoidThreshold


def get_model_type_from_folder(folder_name: str) -> Dict[str, Any]:
    """
    Infer model type and configuration from folder name.

    Returns:
        Dict with 'architecture' and 'encoder' keys
    """
    folder_lower = folder_name.lower()

    if folder_lower.startswith("baseline"):
        return {"architecture": "baseline", "encoder": None}

    elif "dinov3" in folder_lower:
        # Extract encoder size from folder name
        if "vitl" in folder_lower or "dinov3l" in folder_lower:
            encoder = "dinov3_vitl16"
        elif "vitb" in folder_lower or "dinov3b" in folder_lower:
            encoder = "dinov3_vitb16"
        else:  # default to small
            encoder = "dinov3_vits16"
        return {"architecture": "dinov3", "encoder": encoder}

    else:
        raise ValueError(f"Cannot determine model type for: {folder_name}")


def create_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and return the corresponding model."""
    arch = model_config["architecture"]

    if arch == "baseline":
        model = UNet(n_channels=3, n_classes=1, base_ch=32)

    elif arch == "dinov3":
        encoder = model_config["encoder"]
        # Try to infer if encoder was frozen from folder name (optional)
        model = Dinov3UNet(encoder_name=encoder, freeze_encoder=False)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate a model on test set.

    Returns:
        Dictionary with metrics including loss, dice, iou, and inference times
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0
    total_inference_time = 0.0
    num_batches = 0

    # GPU warmup for accurate timing
    if device.type == "cuda":
        warmup_batch = next(iter(dataloader))
        warmup_imgs = warmup_batch["image"].to(device).float()
        _ = model(warmup_imgs)
        torch.cuda.synchronize()
        del warmup_imgs

    # Evaluation loop
    for batch in dataloader:
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()

        # Normalize masks
        if masks.max() > 1.0:
            masks = masks / 255.0
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        elif masks.dim() == 5 and masks.size(1) == 1:
            masks = masks.squeeze(1)

        # Measure inference time
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_inference_time += end_time - start_time
        num_batches += 1

        # Compute loss and metrics
        loss = loss_fn(logits, masks)
        predictions = SigmoidThreshold(logits)

        total_loss += loss.item() * images.size(0)
        total_dice += DiceCoef(predictions, masks).item()
        total_iou += IoU(predictions, masks).item()
        num_samples += images.size(0)

    # Calculate averages
    avg_loss = total_loss / num_samples
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_time_per_batch = (total_inference_time / num_batches) * 1000  # ms
    avg_time_per_sample = (total_inference_time / num_samples) * 1000  # ms

    return {
        "loss": avg_loss,
        "dice": avg_dice,
        "iou": avg_iou,
        "inference_time_per_batch_ms": avg_time_per_batch,
        "inference_time_per_sample_ms": avg_time_per_sample,
        "total_samples": num_samples,
        "num_batches": num_batches,
    }


def load_checkpoint(
    checkpoint_path: Path, model: nn.Module, device: torch.device
) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


def evaluate_all_runs(
    runs_dir: str,
    data_root: str = "data/isic2018",
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 2,
    output_file: str = "evaluation_results.json",
) -> Dict[str, Any]:
    """Evaluate all checkpoints in runs directory."""

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Runs directory: {runs_dir}")
    print(f"Data root: {data_root}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print("=" * 70)

    # Prepare test dataloader
    transforms = T.Compose([T.Resize((image_size, image_size))])
    test_dataset = ag.ISIC2018SkinLesionDataset(
        root=data_root, split="test", transforms=transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Test dataset: {len(test_dataset)} samples")
    print("=" * 70)

    runs_path = Path(runs_dir)
    results = {}

    # Evaluate each run
    for run_folder in sorted(runs_path.iterdir()):
        if not run_folder.is_dir():
            continue

        checkpoint_path = run_folder / "best.pt"
        if not checkpoint_path.exists():
            print(f"\n[SKIP] {run_folder.name}: best.pt not found")
            continue

        print(f"\n[EVAL] {run_folder.name}")
        print("-" * 70)

        try:
            # Determine model type
            model_config = get_model_type_from_folder(run_folder.name)
            print(f"Architecture: {model_config['architecture']}")
            if model_config["encoder"]:
                print(f"Encoder: {model_config['encoder']}")

            # Create and load model
            model = create_model(model_config, device)
            model = load_checkpoint(checkpoint_path, model, device)

            # Evaluate
            metrics = evaluate_model(model, test_loader, device)

            # Store results
            results[run_folder.name] = {
                "model_config": model_config,
                "checkpoint_path": str(checkpoint_path),
                **metrics,
            }

            # Print metrics
            print(f"Loss:              {metrics['loss']:.4f}")
            print(f"Dice:              {metrics['dice']:.4f}")
            print(f"IoU:               {metrics['iou']:.4f}")
            print(f"Time/batch:        {metrics['inference_time_per_batch_ms']:.2f} ms")
            print(
                f"Time/sample:       {metrics['inference_time_per_sample_ms']:.2f} ms"
            )

            # Cleanup
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] {e}")
            results[run_folder.name] = {
                "error": str(e),
                "checkpoint_path": str(checkpoint_path),
            }

    # Save results to JSON
    print("\n" + "=" * 70)
    print("Saving results...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Dice':>8} {'IoU':>8} {'Loss':>8} {'ms/img':>8}")
    print("-" * 70)

    valid_results = {k: v for k, v in results.items() if "error" not in v}

    for name in sorted(valid_results.keys()):
        metrics = valid_results[name]
        print(
            f"{name:<35} "
            f"{metrics['dice']:>8.4f} "
            f"{metrics['iou']:>8.4f} "
            f"{metrics['loss']:>8.4f} "
            f"{metrics['inference_time_per_sample_ms']:>8.2f}"
        )

    # Print errors if any
    error_results = {k: v for k, v in results.items() if "error" in v}
    if error_results:
        print("\n" + "=" * 70)
        print("ERRORS")
        print("=" * 70)
        for name, result in error_results.items():
            print(f"{name}: {result['error']}")

    print("=" * 70)

    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models in runs directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Path to runs directory containing trained models",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/isic2018",
        help="Path to ISIC2018 dataset",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    results = evaluate_all_runs(
        runs_dir=args.runs_dir,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_file=args.output,
    )

    print(f"\nEvaluated {len(results)} runs successfully!")


if __name__ == "__main__":
    main()
