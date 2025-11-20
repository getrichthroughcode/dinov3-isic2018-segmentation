import argparse
from dinoseg.training.train_dino_v3_unet import TrainCfg, TrainDinoUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/isic2018")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--outdir", default="runs/Dinov3Unet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--frozen", type=bool, default=True)
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument(
        "--model",
        default="dinov3_vits16",
        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
    )
    args = ap.parse_args()
    cfg = TrainCfg(
        root=args.root,
        model=args.model,
        size=args.size,
        batch=args.batch,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        outdir=args.outdir,
        seed=args.seed,
        fraction=args.fraction,
        frozen=args.frozen,
        mixed_precision=not args.no_amp,
    )

    TrainDinoUNet(cfg)


if __name__ == "__main__":
    main()
