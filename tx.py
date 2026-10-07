"""
tx.py — Transmitter Module
==========================
Generates random bits, maps them to QPSK or 16-QAM symbols,
applies root-raised-cosine (RRC) pulse shaping, and sets the launch power.

Pipeline:
    generate_bits → modulate → pulse_shape → set_launch_power
"""
import numpy as np
from scipy.signal import upfirdn

# ─── Global signal parameters ─────────────────────────────────────────────────
SYMBOL_RATE_BD = 32e9   # Symbol rate  [Baud]
SPS            = 8      # Samples per symbol
ROLLOFF        = 0.1    # RRC roll-off factor β
SPAN           = 10     # RRC filter span [symbols]; filter length = 2*SPAN*SPS + 1


# ─── Modulation look-up tables ────────────────────────────────────────────────

def _qpsk_table() -> dict:
    """Gray-coded QPSK: 2 bits/symbol, unit average power."""
    s = np.sqrt(2)
    return {
        (0, 0): ( 1 + 1j) / s,
        (0, 1): (-1 + 1j) / s,
        (1, 1): (-1 - 1j) / s,
        (1, 0): ( 1 - 1j) / s,
    }


def _16qam_table() -> dict:
    """
    Gray-coded 16-QAM: 4 bits/symbol, unit average power.

    Bit mapping: (b3 b2 b1 b0)
    b3 b2 → I-axis index (Gray)  →  amplitude level
    b1 b0 → Q-axis index (Gray)  →  amplitude level
    """
    gray_order = [0, 1, 3, 2]               # 2-bit Gray code for indices 0-3
    levels     = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)   # normalised to unit avg power
    table = {}
    for i, gi in enumerate(gray_order):
        for j, gj in enumerate(gray_order):
            bits = ((gi >> 1) & 1, gi & 1, (gj >> 1) & 1, gj & 1)
            table[bits] = complex(levels[i], levels[j])
    return table


_QPSK_TABLE  = _qpsk_table()
_16QAM_TABLE = _16qam_table()

# Constellation arrays exposed for rx.py (demodulation look-up)
QPSK_SYMBOLS  = np.array(list(_QPSK_TABLE.values()),  dtype=complex)
QAM16_SYMBOLS = np.array(list(_16QAM_TABLE.values()), dtype=complex)
QPSK_BITS     = np.array(list(_QPSK_TABLE.keys()),    dtype=np.uint8)
QAM16_BITS    = np.array(list(_16QAM_TABLE.keys()),   dtype=np.uint8)


# ─── Core transmitter functions ───────────────────────────────────────────────

def generate_bits(n_symbols: int,
                  modulation: str = 'QPSK',
                  seed: int = 42) -> np.ndarray:
    """
    Generate a random bit stream for the specified modulation format.

    Parameters
    ----------
    n_symbols : int
        Number of symbols to generate.
    modulation : {'QPSK', '16-QAM'}
        Modulation format.
    seed : int
        NumPy random seed (use different seeds for train / test data).

    Returns
    -------
    bits : np.ndarray, shape (n_symbols * bits_per_symbol,), dtype uint8
    """
    bps = 2 if modulation == 'QPSK' else 4
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)


def modulate(bits: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
    """
    Map a bit stream to complex IQ symbols using the constellation table.

    Parameters
    ----------
    bits : np.ndarray, dtype uint8
        Bit stream; length must be n_symbols × bits_per_symbol.
    modulation : {'QPSK', '16-QAM'}

    Returns
    -------
    symbols : np.ndarray, complex128, shape (n_symbols,)
    """
    if modulation == 'QPSK':
        bps, table = 2, _QPSK_TABLE
    else:
        bps, table = 4, _16QAM_TABLE
    rows = bits.reshape(-1, bps)
    return np.array([table[tuple(r)] for r in rows], dtype=complex)


def rrc_taps(sps: int = SPS,
             rolloff: float = ROLLOFF,
             span: int = SPAN) -> np.ndarray:
    """
    Compute root-raised-cosine (RRC) filter taps.

    The RRC filter is the square root of the raised-cosine (Nyquist) filter.
    Applying the same filter twice (at Tx and Rx) gives the full RC response,
    ensuring zero ISI at integer symbol-period sampling points.

    Parameters
    ----------
    sps     : int   – samples per symbol
    rolloff : float – roll-off factor β ∈ [0, 1]
    span    : int   – filter span in symbols (half-length = span * sps)

    Returns
    -------
    h : np.ndarray, real, length = 2 * span * sps + 1
        Unit-energy RRC impulse response.
    """
    N    = span * sps
    t    = np.arange(-N, N + 1) / sps   # normalised time (symbol periods)
    beta = rolloff
    h    = np.zeros(len(t), dtype=float)

    for k, ti in enumerate(t):
        if abs(ti) < 1e-10:
            # t = 0: limit of the RRC formula
            h[k] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 1e-10 and abs(abs(ti) - 1.0 / (2.0 * beta)) < 1e-6:
            # t = ±1/(2β): second limiting case (wider epsilon for float safety)
            h[k] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[k] = num / den if abs(den) > 1e-10 else 0.0

    h /= np.sqrt(np.sum(h ** 2))        # normalise to unit energy
    return h


def pulse_shape(symbols: np.ndarray,
                sps: int     = SPS,
                rolloff: float = ROLLOFF,
                span: int    = SPAN) -> np.ndarray:
    """
    Upsample by `sps` and apply the RRC transmit filter.

    Parameters
    ----------
    symbols : np.ndarray, complex
        Complex symbol sequence (length N).
    sps     : int   – upsample factor
    rolloff : float – RRC roll-off factor
    span    : int   – RRC filter span in symbols

    Returns
    -------
    signal : np.ndarray, complex
        Pulse-shaped signal at rate sps × Rs  (length ≈ N * sps).
    """
    h = rrc_taps(sps, rolloff, span)
    return upfirdn(h, symbols, up=sps).astype(complex)


def set_launch_power(signal: np.ndarray, power_dBm: float = 0.0) -> np.ndarray:
    """
    Scale signal amplitude so that the average launch power equals `power_dBm`.

    Parameters
    ----------
    signal    : np.ndarray, complex – optical field [√W]
    power_dBm : float               – target average power [dBm]

    Returns
    -------
    signal_scaled : np.ndarray, complex
    """
    power_W = 1e-3 * 10.0 ** (power_dBm / 10.0)
    current  = np.mean(np.abs(signal) ** 2)
    if current < 1e-30:
        return signal
    return signal * np.sqrt(power_W / current)
