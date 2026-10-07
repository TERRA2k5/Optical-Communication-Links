"""
plots.py — Visualization Module
================================
Generates all figures for the project:

    plot_eye_diagram      — overlaid signal traces (open/closed eye)
    plot_constellation    — IQ scatter plots (ideal vs received vs equalized)
    plot_ber_vs_distance  — log-scale BER curve with/without ML
    plot_training_curve   — MSE loss vs epoch

All functions return matplotlib.figure.Figure objects, which are
directly compatible with st.pyplot() in the Streamlit dashboard.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (safe for Streamlit / headless)
import matplotlib.pyplot as plt

# ─── Colour palette (accessible, consistent across all plots) ─────────────────
_C_BLUE   = '#2563EB'   # transmitted / train
_C_RED    = '#DC2626'   # received degraded / no-ML / validation
_C_GREEN  = '#059669'   # equalized / with-ML
_C_ORANGE = '#D97706'   # validation loss / extra
_C_GREY   = '#6B7280'   # reference lines


# ─── Eye diagram ─────────────────────────────────────────────────────────────

def plot_eye_diagram(signal: np.ndarray,
                     sps: int,
                     title: str = 'Eye Diagram',
                     n_traces: int = 200) -> plt.Figure:
    """
    Overlay multiple 2-symbol-period segments to form an eye diagram.

    A well-opened eye indicates low ISI / noise; a closed or distorted eye
    indicates significant impairment.

    Parameters
    ----------
    signal   : np.ndarray, complex – signal at `sps` samples/symbol
    sps      : int   – samples per symbol
    title    : str   – figure title
    n_traces : int   – number of overlaid traces (more → denser eye)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    seg_len    = 2 * sps
    max_traces = min(n_traces, (len(signal) - sps) // sps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, component, label in zip(
            axes,
            [signal.real, signal.imag],
            ['In-Phase (I)', 'Quadrature (Q)']):
        for k in range(max_traces):
            seg = component[k * sps: k * sps + seg_len]
            if len(seg) < seg_len:
                break
            ax.plot(seg, color=_C_BLUE, alpha=0.12, linewidth=0.6)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Sample index within 2T window', fontsize=9)
        ax.set_ylabel('Amplitude  [√W]', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.35)

        # Vertical markers at the ideal sampling instants
        for ts in [sps // 2, sps + sps // 2]:
            ax.axvline(ts, color=_C_GREY, linewidth=0.8, linestyle=':')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


# ─── Constellation diagram ─────────────────────────────────────────────────────

def plot_constellation(symbols_list: list[np.ndarray],
                       labels: list[str],
                       title: str = 'Constellation Diagram',
                       max_points: int = 1000) -> plt.Figure:
    """
    Scatter plot of one or more symbol clouds on the same IQ plane.

    Parameters
    ----------
    symbols_list : list of complex np.ndarray
        E.g. [tx_symbols, rx_symbols, eq_symbols]
    labels       : list of str  – legend labels (same order)
    title        : str
    max_points   : int – cap the number of points per cloud for speed

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colours = [_C_BLUE, _C_RED, _C_GREEN, _C_ORANGE]
    markers = ['o', 's', '^', 'D']
    sizes   = [5,   4,   5,   4]

    fig, ax = plt.subplots(figsize=(6, 6))

    for syms, lbl, col, mk, sz in zip(
            symbols_list, labels, colours, markers, sizes):
        n = min(len(syms), max_points)
        ax.scatter(syms[:n].real, syms[:n].imag,
                   s=sz, alpha=0.55, color=col, marker=mk, label=lbl)

    ax.axhline(0, color=_C_GREY, linewidth=0.7)
    ax.axvline(0, color=_C_GREY, linewidth=0.7)
    ax.set_xlabel('In-Phase (I)', fontsize=11)
    ax.set_ylabel('Quadrature (Q)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=9, loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    return fig


# ─── BER vs distance ──────────────────────────────────────────────────────────

def plot_ber_vs_distance(distances: np.ndarray,
                         bers_noml: np.ndarray,
                         bers_ml:   np.ndarray) -> plt.Figure:
    """
    Log-scale BER vs transmission distance for two scenarios
    (with and without the ML equalizer).

    Parameters
    ----------
    distances : np.ndarray – transmission distances [km]
    bers_noml : np.ndarray – BER without equalizer at each distance
    bers_ml   : np.ndarray – BER with equalizer at each distance

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    floor = 1e-6   # floor for log-scale (avoid log(0))
    bn    = np.clip(bers_noml, floor, 1.0)
    bm    = np.clip(bers_ml,   floor, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(distances, bn, 'o-',
                color=_C_RED,   linewidth=2, markersize=6,
                label='Without ML equalizer')
    ax.semilogy(distances, bm, 's-',
                color=_C_GREEN, linewidth=2, markersize=6,
                label='With ML equalizer')

    ax.axhline(1e-3, color=_C_GREY, linestyle='--', linewidth=1.0,
               label='BER = 10⁻³ threshold')

    ax.set_xlabel('Transmission Distance  [km]', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)',         fontsize=12)
    ax.set_title('BER vs Transmission Distance',  fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig


# ─── Training curve ───────────────────────────────────────────────────────────

def plot_training_curve(history: dict) -> plt.Figure:
    """
    Plot training and validation MSE loss vs epoch.

    Parameters
    ----------
    history : dict with keys 'train_loss' and 'val_loss' (list of floats)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, history['train_loss'],
            color=_C_BLUE,   linewidth=2.0, label='Train loss')
    ax.plot(epochs, history['val_loss'],
            color=_C_ORANGE, linewidth=2.0, linestyle='--', label='Validation loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Neural Network Training Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    return fig


# ─── Utility ──────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a figure to disk and close it to free memory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved → {path}')
