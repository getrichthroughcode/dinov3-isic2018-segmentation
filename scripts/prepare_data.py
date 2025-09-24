import argparse
import any_gold as ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", default="train", choices=["train", "val", "test"])
    ap.add_argument("--override", default="false", choices=["true", "false"])
    args = ap.parse_args()
    _ = ag.ISIC2018SkinLesionDataset(root=args.root, split=s, override=args.override)
    print("ISIC2018 Ready")


if __name__ == "__main__":
    main()
