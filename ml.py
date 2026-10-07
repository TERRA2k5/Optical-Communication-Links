"""
ml.py — Neural-Network Equalizer Module
========================================
Implements a lightweight, window-based fully-connected equalizer that
compensates fibre impairments (dispersion-induced ISI, nonlinear phase
noise) using supervised learning on simulated (received, transmitted)
symbol pairs.

Architecture:
    Input(2W) → FC(64) → ReLU → FC(64) → ReLU → FC(2)

    W = window size (odd number of context symbols centred on current symbol)
    Input:  real and imaginary parts of W consecutive received symbols
    Output: (Re, Im) of the estimated transmitted symbol

Training:
    Loss      : MSE on (Re, Im)            [regression, not classification]
    Optimiser : Adam  (lr = 1e-3)
    Epochs    : 100, batch size 256

Reference: Jarajreh et al. (2015), Freire et al. (2022), Ding et al. (2022)
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
WINDOW   = 7      # context window width (odd); input dim = 2 × WINDOW
HIDDEN   = 64     # hidden layer width
EPOCHS   = 100    # training epochs
BATCH    = 256    # mini-batch size
LR       = 1e-3   # Adam learning rate
VAL_FRAC = 0.2    # fraction of training data reserved for validation


# ─── Model ────────────────────────────────────────────────────────────────────

class NNEqualizer(nn.Module):
    """
    Lightweight fully-connected equalizer.

    Input  : flattened real + imaginary parts of a window of received symbols
             → tensor shape (batch, 2 * window)
    Output : (Re, Im) estimate of the centre transmitted symbol
             → tensor shape (batch, 2)
    """

    def __init__(self, window: int = WINDOW, hidden: int = HIDDEN):
        super().__init__()
        self.window = window
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(2 * window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),        # output: (Re_est, Im_est)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Dataset construction ─────────────────────────────────────────────────────

def build_dataset(rx_symbols: np.ndarray,
                  tx_symbols: np.ndarray,
                  window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a sliding-window (X, Y) dataset from a stream of symbol pairs.

    For each centre index i ∈ [half, N − half):
        X[i] = [Re(rx[i−half:i+half+1]),  Im(rx[i−half:i+half+1])]
        Y[i] = [Re(tx[i]),                 Im(tx[i])]

    Provides ISI context from neighbouring symbols, allowing the NN to
    learn to invert dispersion-induced inter-symbol interference.

    Parameters
    ----------
    rx_symbols : np.ndarray, complex – received symbols after downsampling
    tx_symbols : np.ndarray, complex – corresponding transmitted symbols
    window     : int – must be odd

    Returns
    -------
    X : np.ndarray, float32, shape (n_samples, 2*window)
    Y : np.ndarray, float32, shape (n_samples, 2)
    """
    if window % 2 == 0:
        raise ValueError('window must be odd')

    half = window // 2
    N    = min(len(rx_symbols), len(tx_symbols))
    n    = N - 2 * half

    X = np.empty((n, 2 * window), dtype=np.float32)
    Y = np.empty((n, 2),          dtype=np.float32)

    for i in range(n):
        c      = i + half
        win    = rx_symbols[c - half: c + half + 1]
        X[i]   = np.concatenate([win.real.astype(np.float32),
                                  win.imag.astype(np.float32)])
        Y[i, 0] = float(tx_symbols[c].real)
        Y[i, 1] = float(tx_symbols[c].imag)

    return X, Y


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model: NNEqualizer,
          X_train: np.ndarray,
          Y_train: np.ndarray,
          epochs: int    = EPOCHS,
          batch_size: int = BATCH,
          lr: float       = LR,
          val_frac: float = VAL_FRAC,
          seed: int       = 0) -> dict:
    """
    Train the equalizer with the Adam optimiser and MSE loss.

    The training set is split into train / validation subsets using
    a random permutation with the given seed (reproducibility).

    Parameters
    ----------
    model      : NNEqualizer
    X_train    : np.ndarray, float32, shape (N, 2*W)
    Y_train    : np.ndarray, float32, shape (N, 2)
    epochs     : int
    batch_size : int
    lr         : float – Adam learning rate
    val_frac   : float – fraction of data for validation
    seed       : int

    Returns
    -------
    history : dict with keys 'train_loss' and 'val_loss'
              (lists of per-epoch MSE values)
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Shuffle and split
    n_total = len(X_train)
    n_val   = max(1, int(n_total * val_frac))
    perm    = rng.permutation(n_total)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def _loader(xi, yi, shuffle):
        ds = TensorDataset(torch.from_numpy(xi), torch.from_numpy(yi))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader  = _loader(X_train[tr_idx],  Y_train[tr_idx],  shuffle=True)
    val_loader = _loader(X_train[val_idx], Y_train[val_idx], shuffle=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history   = {'train_loss': [], 'val_loss': []}

    print(f'  Training {len(tr_idx)} samples  |  Validating {n_val} samples')

    for epoch in range(epochs):
        # ── Training step ────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for xb, yb in tr_loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(xb)
        tr_loss = running_loss / len(tr_idx)

        # ── Validation step ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch + 1:3d}/{epochs}  '
                  f'train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}')

    return history


# ─── Inference ────────────────────────────────────────────────────────────────

def equalize(model: NNEqualizer,
             rx_symbols: np.ndarray,
             window: int = WINDOW) -> np.ndarray:
    """
    Apply the trained equalizer to a stream of received symbols.

    Parameters
    ----------
    model      : NNEqualizer – trained model
    rx_symbols : np.ndarray, complex – received symbol sequence (length N)
    window     : int – must match the window used during training

    Returns
    -------
    eq_symbols : np.ndarray, complex
        Equalized symbol estimates.
        Length = N − window + 1  (edges removed to allow full context window).
    """
    model.eval()
    half = window // 2
    N    = len(rx_symbols)
    n    = N - 2 * half

    X = np.empty((n, 2 * window), dtype=np.float32)
    for i in range(n):
        c    = i + half
        win  = rx_symbols[c - half: c + half + 1]
        X[i] = np.concatenate([win.real.astype(np.float32),
                                win.imag.astype(np.float32)])

    with torch.no_grad():
        Y_hat = model(torch.from_numpy(X)).numpy()

    return (Y_hat[:, 0] + 1j * Y_hat[:, 1]).astype(complex)


# ─── Model I/O ────────────────────────────────────────────────────────────────

def save_model(model: NNEqualizer, path: str = 'models/equalizer.pt') -> None:
    """Save model weights and hyper-parameters to a .pt checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'window'    : model.window,
        'hidden'    : model.hidden,
    }, path)
    print(f'  Model saved → {path}')


def load_model(path: str = 'models/equalizer.pt') -> NNEqualizer:
    """Load model from a checkpoint saved by save_model()."""
    ckpt  = torch.load(path, map_location='cpu', weights_only=True)
    model = NNEqualizer(window=ckpt['window'], hidden=ckpt['hidden'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model
