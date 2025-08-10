from __future__ import annotations
import os, glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import SimpleITK as sitk
from skimage.transform import resize


def _clamp_and_normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clamp HU to [lo, hi] then min-max normalize to [0, 1]."""
    x = np.clip(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-8)
    return x.astype(np.float32)


def _mask_to_tumor_binary(m: np.ndarray) -> np.ndarray:
    """LiTS: 0=bg, 1=liver, 2=tumor -> return tumor mask only (0/1)."""
    return (m == 2).astype(np.float32)


def _resize2d(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize with linear interpolation (preserve_range=True keeps numeric scale)."""
    if arr.shape != out_hw:
        arr = resize(arr, out_hw, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return arr


class LiTSSliceDataset(Dataset):
    """Reads NIfTI volumes and exposes them as 2D slices."""

    def __init__(
        self,
        root_raw: str,
        clamp: Tuple[float, float] = (-45.0, 167.0),
        input_size: Tuple[int, int] = (512, 512),
        cache_index: bool = True,
        slice_filter: Optional[str] = None,  # "tumor_only" to keep only slices containing tumor
    ) -> None:
        super().__init__()
        self.root_raw = root_raw
        self.clamp = clamp
        self.input_size = input_size

        imgs = sorted(glob.glob(os.path.join(root_raw, "images", "*.nii*")))
        msks = sorted(glob.glob(os.path.join(root_raw, "masks", "*.nii*")))
        assert imgs and len(imgs) == len(msks), "Missing or unmatched images/masks under data/raw/"

        def stem(p: str) -> str:
            b = os.path.basename(p)
            return b.replace(".nii.gz", "").replace(".nii", "")

        m_map = {stem(p): p for p in msks}
        self.pairs: List[Tuple[str, str]] = [(ip, m_map[stem(ip)]) for ip in imgs if stem(ip) in m_map]
        assert self.pairs, "No matching image/mask basenames found."

        # Pre-build slice index for fast sampling
        self.index: List[Tuple[int, int]] = []
        for vidx, (ip, mp) in enumerate(self.pairs):
            img = sitk.GetArrayFromImage(sitk.ReadImage(ip))  # [D,H,W]
            msk = sitk.GetArrayFromImage(sitk.ReadImage(mp))
            assert img.shape == msk.shape, f"Shape mismatch: {ip} vs {mp}"
            for s in range(img.shape[0]):
                if slice_filter == "tumor_only" and (msk[s] == 2).sum() == 0:
                    continue
                self.index.append((vidx, s))

    def __len__(self) -> int:
        return len(self.index)

    def _read_slice(self, vidx: int, sidx: int) -> Tuple[np.ndarray, np.ndarray]:
        ip, mp = self.pairs[vidx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(ip))  # [D,H,W]
        msk = sitk.GetArrayFromImage(sitk.ReadImage(mp))
        return img[sidx].astype(np.float32), msk[sidx].astype(np.int16)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        vidx, sidx = self.index[i]
        img_s, msk_s = self._read_slice(vidx, sidx)

        # intensity + mask processing
        img_s = _clamp_and_normalize(img_s, *self.clamp)
        msk_s = _mask_to_tumor_binary(msk_s)

        # resize to target H, W
        H, W = self.input_size
        img_s = _resize2d(img_s, (H, W))
        msk_s = _resize2d(msk_s, (H, W))

        # add channel dim -> [1, H, W]
        img_t = torch.from_numpy(img_s)[None, ...]
        msk_t = torch.from_numpy(msk_s)[None, ...]
        return {"image": img_t, "mask": msk_t}


def make_dataloaders(
    root_raw: str,
    clamp: Tuple[float, float],
    input_size: Tuple[int, int],
    split: Dict[str, float],
    batch_size: int,
    num_workers: int = 4,
    seed: int = 42,
    tumor_only_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with a reproducible split."""
    ds = LiTSSliceDataset(
        root_raw=root_raw,
        clamp=clamp,
        input_size=input_size,
        slice_filter=("tumor_only" if tumor_only_train else None),
    )

    tr = split.get("train", 0.8)
    va = split.get("val", 0.1)
    te = split.get("test", 0.1)
    assert abs(tr + va + te - 1.0) < 1e-6, "split ratios must sum to 1."

    lengths = [int(len(ds) * tr), int(len(ds) * va)]
    lengths.append(len(ds) - sum(lengths))
    g = torch.Generator().manual_seed(seed)
    ds_train, ds_val, ds_test = random_split(ds, lengths, generator=g)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test
