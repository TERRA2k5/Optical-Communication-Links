"""
rx.py — Receiver Module
=======================
Full coherent receiver chain:

    Received signal
        → CDC equalizer            (frequency-domain chromatic dispersion compensation)
        → Matched RRC filter       (cancels ISI, maximises SNR)
        → Downsample               (1 sample per symbol)
        → Phase correction         (M-th power carrier recovery)
        → Hard-decision demod      (nearest constellation point)
        → BER computation

The matched filter uses the same RRC taps as the transmitter, so the
combined Tx–Rx filter is a raised-cosine (Nyquist) response with zero ISI
at integer symbol-period sampling points in the back-to-back case.

Chromatic dispersion compensation (CDC) is applied in the frequency domain
before the matched filter. It inverts the accumulated dispersion of all spans.
"""
import numpy as np
from scipy.signal import upfirdn

from tx import (rrc_taps, SPS, ROLLOFF, SPAN,
                QPSK_SYMBOLS, QAM16_SYMBOLS,
                QPSK_BITS,   QAM16_BITS,
                SYMBOL_RATE_BD)
from fiber import _beta2_s2_km, DEFAULT_FIBER


# ─── Chromatic Dispersion Compensation ───────────────────────────────────────

def chromatic_dispersion_compensate(signal: np.ndarray,
                                    total_length_km: float,
                                    fs: float,
                                    fiber_params: dict = None) -> np.ndarray:
    """
    Frequency-domain chromatic dispersion compensation (CDC).

    Applies the inverse of the accumulated linear dispersion transfer function:
        H_CDC(ω) = exp(+j · β₂/2 · ω² · L_total)

    This exactly cancels the dispersive phase accumulated over L_total km
    of fibre (ignoring nonlinear effects, which are small for the signals here).

    Parameters
    ----------
    signal        : np.ndarray, complex – received waveform at rate fs [√W]
    total_length_km : float – total fibre length to compensate [km]
    fs            : float  – sampling rate [Hz]
    fiber_params  : dict   – fiber parameter dict (uses DEFAULT_FIBER if None)

    Returns
    -------
    compensated : np.ndarray, complex – CDC-equalised signal (same length)
    """
    if fiber_params is None:
        fiber_params = DEFAULT_FIBER

    beta2 = _beta2_s2_km(fiber_params['D_ps_nm_km'],
                          fiber_params['wavelength_nm'])   # s²/km

    N     = len(signal)
    omega = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / fs)   # rad/s

    # Inverse dispersion filter: conjugate of the forward dispersive phase
    H_cdc = np.exp(1j * beta2 / 2.0 * omega ** 2 * total_length_km)

    return np.fft.ifft(np.fft.fft(signal) * H_cdc)


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

def phase_correct(rx_symbols: np.ndarray,
                  modulation: str = 'QPSK',
                  tx_symbols: np.ndarray = None) -> np.ndarray:
    """
    Carrier phase recovery and rotational ambiguity resolution.

    When reference `tx_symbols` are available (as in coherent pilot tracking,
    DSP training frames, or simulation), the exact carrier phase offset θ
    accumulated through fiber propagation is estimated via pilot correlation:
        θ = arg( Σ r_k · s_k* )
    and derotated: r_corrected = r · exp(-jθ).

    When `tx_symbols` is None, blind carrier phase recovery is applied:
    4th-power Viterbi-Viterbi for QPSK, or minimum-distance quadrant search.

    Parameters
    ----------
    rx_symbols : np.ndarray, complex – received symbols
    modulation : {'QPSK', '16-QAM'}
    tx_symbols : np.ndarray, complex (optional) – known reference/pilot symbols

    Returns
    -------
    symbols_corrected : np.ndarray, complex – phase-aligned symbols
    """
    ref = QPSK_SYMBOLS if modulation == 'QPSK' else QAM16_SYMBOLS

    # ── Reference / Pilot-aided carrier phase alignment ───────────────────────
    if tx_symbols is not None and len(tx_symbols) > 0:
        n = min(len(rx_symbols), len(tx_symbols))
        # Use initial pilot block (up to 64 symbols) to lock the exact carrier angle
        n_pilot = min(n, 64)
        corr = np.mean(rx_symbols[:n_pilot] * np.conj(tx_symbols[:n_pilot]))
        if abs(corr) > 1e-15:
            theta = np.angle(corr)
            return rx_symbols * np.exp(-1j * theta)
        return rx_symbols

    # ── Blind carrier phase recovery ─────────────────────────────────────────
    if modulation == 'QPSK':
        # 4th-power Viterbi-Viterbi estimation
        avg_4th = np.mean(rx_symbols ** 4)
        theta_est = 0.25 * (np.angle(-avg_4th))
        derotated = rx_symbols * np.exp(-1j * theta_est)
    else:
        derotated = rx_symbols

    best_syms  = derotated
    best_score = np.inf

    for k in range(4):
        # Resolve 4-fold ambiguity: 0°, 90°, 180°, 270°
        rotated = derotated * np.exp(1j * k * np.pi / 2.0)
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
            span: int = SPAN,
            total_length_km: float = 0.0,
            fs: float = None,
            fiber_params: dict = None) -> dict:
    """
    Run the full receiver pipeline on a received signal.

    Steps:  CDC → MF → downsample → normalise → phase_correct → demodulate → BER

    Parameters
    ----------
    signal          : np.ndarray, complex – received optical field [√W]
    tx_symbols      : np.ndarray, complex – transmitted symbols (for alignment)
    tx_bits         : np.ndarray, uint8   – transmitted bits (for BER)
    modulation      : {'QPSK', '16-QAM'}
    sps             : int   – samples per symbol
    rolloff         : float – RRC roll-off factor
    span            : int   – RRC filter span in symbols
    total_length_km : float – total fibre length for CDC; 0 = no CDC (back-to-back)
    fs              : float – sampling rate [Hz]; required when total_length_km > 0
    fiber_params    : dict  – fiber params for CDC; uses DEFAULT_FIBER if None

    Returns
    -------
    result : dict with keys
        'rx_symbols'     – phase-corrected, normalised received symbols
        'rx_symbols_raw' – after normalise, before phase correction
        'rx_bits'        – demodulated bits
        'ber'            – bit error rate
        'mf_signal'      – matched-filter output (for eye diagram)
    """
    bps = 2 if modulation == 'QPSK' else 4

    # ── Step 1: Chromatic Dispersion Compensation ─────────────────────────────
    if total_length_km > 0.0 and fs is not None:
        sig_cdc = chromatic_dispersion_compensate(
            signal, total_length_km, fs, fiber_params)
    else:
        sig_cdc = signal

    # ── Step 2: Matched filter + downsample ───────────────────────────────────
    mf_out     = matched_filter(sig_cdc, sps, rolloff, span)
    rx_sym_raw = downsample(mf_out, sps)

    # Align lengths (fibre simulation may add/remove a few samples at edges)
    n          = min(len(rx_sym_raw), len(tx_symbols))
    rx_sym_raw = rx_sym_raw[:n]
    tx_sym     = tx_symbols[:n]
    tx_b       = tx_bits[:n * bps]

    # ── Step 3: Normalise to unit average power ───────────────────────────────
    # Required so received symbols sit on the same scale as the ideal
    # constellation, enabling correct demodulation and NN equalization.
    rms = np.sqrt(np.mean(np.abs(rx_sym_raw) ** 2))
    if rms > 1e-15:
        rx_sym_raw = rx_sym_raw / rms

    # ── Step 4: Phase correction, demodulation, BER ───────────────────────────
    rx_sym  = phase_correct(rx_sym_raw, modulation, tx_symbols=tx_sym)
    rx_bits = demodulate(rx_sym, modulation)
    ber     = compute_ber(tx_b, rx_bits)

    return {
        'rx_symbols'     : rx_sym,
        'rx_symbols_raw' : rx_sym_raw,
        'rx_bits'        : rx_bits,
        'ber'            : ber,
        'mf_signal'      : mf_out,
    }
