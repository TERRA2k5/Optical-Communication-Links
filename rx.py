"""
rx.py — Receiver Module
=======================
Full coherent receiver chain:

    Received signal
        → Matched RRC filter       (cancels ISI, maximises SNR)
        → Downsample               (1 sample per symbol)
        → Phase correction         (M-th power carrier recovery)
        → Hard-decision demod      (nearest constellation point)
        → BER computation

The matched filter uses the same RRC taps as the transmitter, so the
combined Tx–Rx filter is a raised-cosine (Nyquist) response with zero ISI
at integer symbol-period sampling points in the back-to-back case.
"""
import numpy as np
from scipy.signal import upfirdn

from tx import (rrc_taps, SPS, ROLLOFF, SPAN,
                QPSK_SYMBOLS, QAM16_SYMBOLS,
                QPSK_BITS,   QAM16_BITS)


# ─── Matched filter ───────────────────────────────────────────────────────────

def matched_filter(signal: np.ndarray,
                   sps: int     = SPS,
                   rolloff: float = ROLLOFF,
                   span: int    = SPAN) -> np.ndarray:
    """
    Apply the RRC matched filter (identical taps to Tx filter).

    The combined Tx × Rx response is a raised-cosine filter, ensuring
    zero ISI at the symbol sampling instants in the ideal (no-fibre) case.

    Parameters
    ----------
    signal  : np.ndarray, complex – received signal at rate sps × Rs [√W]
    sps     : int   – samples per symbol (must match Tx)
    rolloff : float – RRC roll-off factor (must match Tx)
    span    : int   – RRC filter span in symbols (must match Tx)

    Returns
    -------
    filtered : np.ndarray, complex – filtered signal (same rate as input)
    """
    h = rrc_taps(sps, rolloff, span)
    return upfirdn(h, signal, up=1, down=1).astype(complex)


# ─── Downsampling ─────────────────────────────────────────────────────────────

def downsample(signal: np.ndarray,
               sps: int = SPS,
               offset: int = None) -> np.ndarray:
    """
    Trim filter delay and downsample to 1 sample per symbol.

    The total group delay introduced by both RRC filters (Tx + Rx) is
        delay = 2 × SPAN × sps   samples
    This is trimmed from the head of the signal before selecting every
    sps-th sample.

    The sampling offset within a symbol period is chosen automatically
    as the index that maximises the mean received power (open eye).

    Parameters
    ----------
    signal : np.ndarray, complex – matched-filter output
    sps    : int   – samples per symbol
    offset : int or None – sampling offset within [0, sps); auto-selected if None

    Returns
    -------
    symbols : np.ndarray, complex – downsampled symbol sequence
    """
    delay = 2 * SPAN * sps          # total group delay of both RRC filters
    sig   = signal[delay:]           # trim head (skip transient)

    if len(sig) < sps:
        return sig[:1]

    # Auto-select the sampling offset that maximises eye-opening
    if offset is None:
        eye_power = np.array([
            np.mean(np.abs(sig[k::sps]) ** 2) for k in range(sps)
        ])
        offset = int(np.argmax(eye_power))

    return sig[offset::sps]


# ─── Phase correction ─────────────────────────────────────────────────────────

def phase_correct(rx_symbols: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
    """
    Resolve the 4-fold (π/2) rotational ambiguity by trying all four
    candidate rotations {0°, 90°, 180°, 270°} and selecting the one that
    minimises the total squared distance to the reference constellation.

    In this simulation there is no RF carrier phase offset (the SSFM and
    EDFA do not introduce a systematic constant phase rotation), so the
    optimal rotation is typically 0°.  The search handles any edge case
    where the receiver chain introduces a 90° multiple offset.

    Parameters
    ----------
    rx_symbols : np.ndarray, complex – received symbols
    modulation : {'QPSK', '16-QAM'}

    Returns
    -------
    symbols_corrected : np.ndarray, complex – best-rotation-aligned symbols
    """
    ref = QPSK_SYMBOLS if modulation == 'QPSK' else QAM16_SYMBOLS
    best_syms  = rx_symbols
    best_score = np.inf

    for k in range(4):
        # Try rotations 0°, 90°, 180°, 270°
        rotated = rx_symbols * np.exp(1j * k * np.pi / 2.0)
        dists   = np.min(np.abs(rotated[:, None] - ref[None, :]) ** 2, axis=1)
        score   = float(np.sum(dists))
        if score < best_score:
            best_score = score
            best_syms  = rotated

    return best_syms


# ─── Hard-decision demodulation ───────────────────────────────────────────────

def demodulate(symbols: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
    """
    Map each received symbol to the nearest constellation point (hard decision)
    and return the corresponding bit sequence.

    Parameters
    ----------
    symbols    : np.ndarray, complex
    modulation : {'QPSK', '16-QAM'}

    Returns
    -------
    bits : np.ndarray, uint8 – length = n_symbols × bits_per_symbol
    """
    if modulation == 'QPSK':
        ref_syms = QPSK_SYMBOLS
        ref_bits = QPSK_BITS
    else:
        ref_syms = QAM16_SYMBOLS
        ref_bits = QAM16_BITS

    # Distance matrix: (n_symbols, n_constellation_points)
    idx  = np.argmin(np.abs(symbols[:, None] - ref_syms[None, :]) ** 2, axis=1)
    return ref_bits[idx].flatten()


# ─── BER ──────────────────────────────────────────────────────────────────────

def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Compute the bit error rate.

    Parameters
    ----------
    tx_bits : np.ndarray, uint8 – transmitted bits
    rx_bits : np.ndarray, uint8 – detected bits

    Returns
    -------
    ber : float ∈ [0, 1]
    """
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 1.0
    errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    return errors / n


# ─── Full receiver chain ──────────────────────────────────────────────────────

def receive(signal: np.ndarray,
            tx_symbols: np.ndarray,
            tx_bits: np.ndarray,
            modulation: str = 'QPSK',
            sps: int = SPS,
            rolloff: float = ROLLOFF,
            span: int = SPAN) -> dict:
    """
    Run the full receiver pipeline on a received signal.

    Steps:  MF → downsample → phase_correct → demodulate → BER

    Parameters
    ----------
    signal     : np.ndarray, complex – received optical field [√W]
    tx_symbols : np.ndarray, complex – transmitted symbols (for length alignment)
    tx_bits    : np.ndarray, uint8   – transmitted bits (for BER computation)
    modulation : {'QPSK', '16-QAM'}
    sps        : int   – samples per symbol
    rolloff    : float – RRC roll-off factor
    span       : int   – RRC filter span in symbols

    Returns
    -------
    result : dict with keys
        'rx_symbols'     – phase-corrected received symbols
        'rx_symbols_raw' – before phase correction
        'rx_bits'        – demodulated bits
        'ber'            – bit error rate
        'mf_signal'      – matched-filter output (for eye diagram)
    """
    bps = 2 if modulation == 'QPSK' else 4

    mf_out     = matched_filter(signal, sps, rolloff, span)
    rx_sym_raw = downsample(mf_out, sps)

    # Align lengths (fibre simulation may add/remove a few samples at edges)
    n          = min(len(rx_sym_raw), len(tx_symbols))
    rx_sym_raw = rx_sym_raw[:n]
    tx_sym     = tx_symbols[:n]
    tx_b       = tx_bits[:n * bps]

    rx_sym     = phase_correct(rx_sym_raw, modulation)
    rx_bits    = demodulate(rx_sym, modulation)
    ber        = compute_ber(tx_b, rx_bits)

    return {
        'rx_symbols'     : rx_sym,
        'rx_symbols_raw' : rx_sym_raw,
        'rx_bits'        : rx_bits,
        'ber'            : ber,
        'mf_signal'      : mf_out,
    }
