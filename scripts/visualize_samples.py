import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import argparse
import math
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from utils.viz import OverlayMask, MakeGrid, SaveGrid

import any_gold as ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/isic2018")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--size", default=256)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--outdir", dafault="results/samples")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_arg()

    torch.manual_seed(args.seed)
    ds = ag.ISIC2018SkinLesionDataset(
        root=args.root, split=args.split, transforms=T.Compose([T.Resize((args.size, args.size))])
    )
    dl = DatalLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    batch = next(iter(dl))
    imgs = batch["image"]
    masks = batch["masks"]
    k = min(args.num, imgs.size(0))
    overlays = [OverlayMask(imgs[i], masks[i]) for i in range(k)]
    nrow = int(math.sqrt(k)) or 1
    grid = MakeGrid(overlays, nrow)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"isic2018_{args.split}_{args.size}_{k}.png")
    SaveGrid(grid, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
