from __future__ import annotations
import argparse, yaml, torch
from tqdm import tqdm

from .dataset import make_dataloaders
from .unet import UNet
from .metrics import dice_coef, iou_coef


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str, help="Path to best.pth")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Only need the test loader here
    _, _, dl_test = make_dataloaders(
        root_raw=cfg["data"]["root_raw"],
        clamp=tuple(cfg["data"].get("clamp", [-45, 167])),
        input_size=tuple(cfg["data"].get("input_size", [512, 512])),
        split=cfg["data"].get("split", {"train": 0.8, "val": 0.1, "test": 0.1}),
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        seed=int(cfg.get("seed", 42)),
        tumor_only_train=bool(cfg["data"].get("tumor_only_train", False)),
    )

    model = UNet(in_channels=1, out_channels=1, base=64).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dice, iou = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Evaluate(Test)"):
            imgs = batch["image"].to(device)
            msks = batch["mask"].to(device)
            logits = model(imgs)
            dice += dice_coef(logits, msks)
            iou  += iou_coef(logits, msks)

    dice /= max(len(dl_test), 1)
    iou  /= max(len(dl_test), 1)
    print(f"[EVAL] Dice={dice:.4f} | IoU={iou:.4f}")


if __name__ == "__main__":
    main()
