import argparse
import json
import time
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


def get_model_type_from_folder(folder_name: str) -> str:
    """Infer model type from folder name."""
    folder_lower = folder_name.lower()
    if folder_lower.startswith("baseline"):
        return "baseline"
    elif "dinov3b" in folder_lower:
        return "dinov3b"
    elif "dinov3l" in folder_lower:
        return "dinov3l"
    elif "dinov3s" in folder_lower:
        return "dinov3s"
    elif "dinov2" in folder_lower:
        return "dinov2"
    else:
        raise ValueError(f"Cannot determine model type for: {folder_name}")


def create_model(model_name: str, device: torch.device) -> nn.Module:
    """Create and return the corresponding model."""
    if model_name == "baseline":
        return UNet(n_channels=3, n_classes=1, base_ch=32).to(device)
    elif model_name == "dinov2":
        return DinoUNet(n_classes=1, encoder_name="dinov2_vits14").to(device)
    elif model_name == "dinov3s":
        return Dinov3UNet(n_classes=1, encoder_name="dinov3_vits16").to(device)
    elif model_name == "dinov3b":
        return Dinov3UNet(n_classes=1, encoder_name="dinov3_vitb16").to(device)
    elif model_name == "dinov3l":
        return Dinov3UNet(n_classes=1, encoder_name="dinov3_vitl16").to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


@torch.no_grad()
def evaluate_single(
    model: nn.Module,
    dl_test: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a model and return metrics including inference time."""
    loss_fn = nn.BCEWithLogitsLoss()
    tot_loss, tot_dice, tot_iou = 0.0, 0.0, 0.0
    n_samples = 0
    total_inference_time = 0.0
    n_batches = 0

    # GPU warmup for accurate timing
    if device.type == "cuda":
        for batch in dl_test:
            imgs = batch["image"].to(device).float()
            _ = model(imgs)
            break
        torch.cuda.synchronize()

    for batch in dl_test:
        imgs = batch["image"].to(device).float()
        masks = batch["mask"].to(device).float()

        if masks.max() > 1.0:
            masks = masks / 255.0
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        # Measure inference time
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        logits = model(imgs)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_inference_time += end_time - start_time
        n_batches += 1

        loss = loss_fn(logits, masks)
        preds = SigmoidThreshold(logits)

        tot_loss += loss.item() * imgs.size(0)
        tot_dice += DiceCoef(preds, masks)
        tot_iou += IoU(preds, masks)
        n_samples += imgs.size(0)

    avg_loss = tot_loss / n_samples
    avg_dice = tot_dice / len(dl_test)
    avg_iou = tot_iou / len(dl_test)

    # Convert to float if tensor
    if hasattr(avg_dice, "item"):
        avg_dice = avg_dice.item()
    if hasattr(avg_iou, "item"):
        avg_iou = avg_iou.item()
    avg_inference_time_per_batch = (total_inference_time / n_batches) * 1000
    avg_inference_time_per_sample = (total_inference_time / n_samples) * 1000

    return {
        "loss": avg_loss,
        "dice": avg_dice,
        "iou": avg_iou,
        "inference_time_per_batch_ms": avg_inference_time_per_batch,
        "inference_time_per_sample_ms": avg_inference_time_per_sample,
        "total_samples": n_samples,
    }


def evaluate_all(
    runs_dir: str,
    input_size: int = 256,
    batch_size: int = 8,
    data_root: str = "data/isic2018",
    output_file: str = "evaluation_results.json",
):
    """Evaluate all best.pt models in the runs directory."""

    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print("=" * 60)

    # Prepare dataloader (shared across all models)
    transforms = T.Compose([T.Resize((input_size, input_size))])
    ds_test = ag.ISIC2018SkinLesionDataset(
        root=data_root, split="test", transforms=transforms
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Dataset loaded: {len(ds_test)} test samples")
    print("=" * 60)

    runs_path = Path(runs_dir)
    results = {}

    # Iterate over all subfolders
    for run_folder in sorted(runs_path.iterdir()):
        if not run_folder.is_dir():
            continue

        best_pt = run_folder / "best.pt"
        if not best_pt.exists():
            print(f"[WARN] {run_folder.name}: best.pt not found, skipping")
            continue

        print(f"\n[EVAL] {run_folder.name}")
        print("-" * 40)

        try:
            # Infer model type
            model_type = get_model_type_from_folder(run_folder.name)
            print(f"   Model type: {model_type}")

            # Load model
            model = create_model(model_type, device)
            ckpt = torch.load(best_pt, map_location=device)

            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)

            model.eval()

            # Evaluate
            metrics = evaluate_single(model, dl_test, device)

            results[run_folder.name] = {
                "model_type": model_type,
                "weights_path": str(best_pt),
                **metrics,
            }

            print(f"   Loss:  {metrics['loss']:.4f}")
            print(f"   Dice:  {metrics['dice']:.4f}")
            print(f"   IoU:   {metrics['iou']:.4f}")
            print(
                f"   Inference time/batch: {metrics['inference_time_per_batch_ms']:.2f} ms"
            )
            print(
                f"   Inference time/sample: {metrics['inference_time_per_sample_ms']:.2f} ms"
            )

            # Free memory
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   [ERROR] {e}")
            results[run_folder.name] = {"error": str(e)}

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Model':<30} {'Dice':>8} {'IoU':>8} {'Loss':>8} {'ms/sample':>10}")
    print("-" * 70)

    for name, metrics in sorted(results.items()):
        if "error" in metrics:
            print(f"{name:<30} {'ERROR':>8}")
        else:
            print(
                f"{name:<30} "
                f"{metrics['dice']:>8.4f} "
                f"{metrics['iou']:>8.4f} "
                f"{metrics['loss']:>8.4f} "
                f"{metrics['inference_time_per_sample_ms']:>10.2f}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all best.pt models in a runs folder"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Path to the runs directory (default: runs)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/isic2018",
        help="Path to the data (default: data/isic2018)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Input image size (default: 256)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file (default: evaluation_results.json)",
    )

    args = parser.parse_args()

    evaluate_all(
        runs_dir=args.runs_dir,
        input_size=args.size,
        batch_size=args.batch,
        data_root=args.data_root,
        output_file=args.output,
    )
