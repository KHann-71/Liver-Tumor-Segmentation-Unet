from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict

import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import make_dataloaders
from .unet import UNet
from .losses import dice_loss, BCEDiceLoss
from .metrics import dice_coef, iou_coef


def set_seed(seed: int = 42) -> None:
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(p: str) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Data
    dl_train, dl_val, dl_test = make_dataloaders(
        root_raw=cfg["data"]["root_raw"],
        clamp=tuple(cfg["data"].get("clamp", [-45, 167])),
        input_size=tuple(cfg["data"].get("input_size", [512, 512])),
        split=cfg["data"].get("split", {"train": 0.8, "val": 0.1, "test": 0.1}),
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        seed=int(cfg.get("seed", 42)),
        tumor_only_train=bool(cfg["data"].get("tumor_only_train", False)),
    )

    # Model
    model = UNet(in_channels=1, out_channels=1, base=64).to(device)

    # Loss + Optim
    loss_name = cfg["train"].get("loss", "dice").lower()
    criterion = BCEDiceLoss(cfg["train"].get("bce_weight", 0.5)) if loss_name == "bce_dice" else dice_loss
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    # Output directory
    out_dir = Path(cfg.get("out_dir", "experiments/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    writer = SummaryWriter(log_dir=str(out_dir / "logs"))
    best_dice = -1.0
    history = []

    # ================= TRAINING LOOP =================
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{cfg['train']['epochs']} [train]")

        for batch in pbar:
            imgs = batch["image"].to(device)  # [B,1,H,W]
            msks = batch["mask"].to(device)   # [B,1,H,W]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)
                loss = criterion(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(dl_train.dataset)

        # -------- Validation --------
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"Epoch {epoch} [val]"):
                imgs = batch["image"].to(device)
                msks = batch["mask"].to(device)
                logits = model(imgs)
                l = criterion(logits, msks) if callable(criterion) else criterion(logits, msks)
                val_loss += l.item() * imgs.size(0)
                val_dice += dice_coef(logits, msks)
                val_iou  += iou_coef(logits, msks)

        val_loss /= max(len(dl_val.dataset), 1)
        val_dice /= max(len(dl_val), 1)
        val_iou  /= max(len(dl_val), 1)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,  epoch)
        writer.add_scalar("metrics/dice", val_dice, epoch)
        writer.add_scalar("metrics/iou",  val_iou,  epoch)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou})
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # save best checkpoint by val dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict()}, out_dir / "best.pth")

    writer.close()

    # -------- Optional: quick test after training --------
    if (out_dir / "best.pth").exists() and len(dl_test) > 0:
        ckpt = torch.load(out_dir / "best.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        test_dice, test_iou = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(dl_test, desc="Test"):
                imgs = batch["image"].to(device)
                msks = batch["mask"].to(device)
                logits = model(imgs)
                test_dice += dice_coef(logits, msks)
                test_iou  += iou_coef(logits, msks)
        test_dice /= len(dl_test)
        test_iou  /= len(dl_test)
        print(f"[TEST] Dice={test_dice:.4f} | IoU={test_iou:.4f}")


if __name__ == "__main__":
    main()
