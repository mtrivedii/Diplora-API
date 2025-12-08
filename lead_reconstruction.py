import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# lead_reconstruction.py

class LeadReconstructionDataset(Dataset):
    """
    ...
    """

    def __init__(self, preproc_dir: str, fixed_len: int):
        self.fixed_len = int(fixed_len)
        pattern = os.path.join(preproc_dir, "file_*.npy")
        files = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No preprocessed files found in {preproc_dir}")
        self.files: List[str] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr = np.load(self.files[idx])  # (12, T)
        if arr.ndim != 2 or arr.shape[0] != 12:
            raise ValueError(f"Expected (12, T), got {arr.shape}")

        # --- crop/pad to fixed length ---
        T = arr.shape[1]
        L = self.fixed_len
        if T > L:
            # center crop
            start = (T - L) // 2
            arr = arr[:, start:start + L]
        elif T < L:
            # pad with zeros at the end
            padded = np.zeros((12, L), dtype=arr.dtype)
            padded[:, :T] = arr
            arr = padded
        # ---------------------------------

        x = arr[[0, 1, 7], :]   # I, II, V2
        y = arr                 # full 12‑lead

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

class LeadReconstructionNet(nn.Module):
    """Simple 1D CNN mapping 3 leads → 12 leads."""

    def __init__(self, base_channels: int = 64, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(3, base_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, 12, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_reconstruction(
    preproc_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    fixed_len: int = 5000,
) -> LeadReconstructionNet:
    """
    Train LeadReconstructionNet on preprocessed files in preproc_dir.

    Returns the trained model (on CPU).
    """
    ds = LeadReconstructionDataset(preproc_dir, fixed_len=fixed_len)
    n = len(ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = LeadReconstructionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.L1Loss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            y_hat = model(x)
            loss = crit(y_hat, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val", leave=False):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = crit(y_hat, y)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    return model
