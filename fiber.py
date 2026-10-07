"""
fiber.py — Fiber Propagation and EDFA Module
=============================================
Solves the Nonlinear Schrödinger Equation (NLSE) using the symmetric
Split-Step Fourier Method (SSFM):

    ∂A/∂z = −(α/2)A − j(β₂/2) ∂²A/∂t² + jγ|A|²A

where
    α  = power attenuation coefficient   [neper/km]
    β₂ = group-velocity dispersion       [s²/km]
    γ  = Kerr nonlinear coefficient      [W⁻¹ km⁻¹]

After each span the EDFA restores the signal power and adds
amplified spontaneous emission (ASE) noise.
"""
import numpy as np

# ─── Physical constants ───────────────────────────────────────────────────────
_H_PLANCK = 6.626e-34   # Planck constant [J·s]
_C_LIGHT  = 3.0e8       # Speed of light  [m/s]

# ─── Default parameter dictionaries ──────────────────────────────────────────
DEFAULT_FIBER = {
    'alpha_dB_km'   : 0.2,      # Attenuation         [dB/km]
    'D_ps_nm_km'    : 17.0,     # Chromatic dispersion [ps/(nm·km)]
    'gamma_W_km'    : 1.3,      # Nonlinear coeff      [W⁻¹ km⁻¹]
    'wavelength_nm' : 1550.0,   # Carrier wavelength   [nm]
}

DEFAULT_EDFA = {
    'noise_figure_dB': 5.0,     # Noise figure         [dB]
}


# ─── Unit-conversion helpers ──────────────────────────────────────────────────

def _alpha_power_km(alpha_dB_km: float) -> float:
    """
    Convert fibre attenuation from dB/km to power-decay neper/km.
        P(z) = P(0) · exp(−α_neper · z)
        α_neper = α_dB · ln(10) / 10
    The NLSE uses α_neper/2 for the field (amplitude) decay.
    """
    return alpha_dB_km * np.log(10.0) / 10.0   # neper/km (power coefficient)


def _beta2_s2_km(D_ps_nm_km: float, wavelength_nm: float) -> float:
    """
    Convert dispersion parameter D [ps/(nm·km)] to β₂ [s²/km].

    Derivation:
        D [ps/(nm·km)]  =  D × 10⁻¹²/(10⁻⁹ × 10³) s/m²  =  D × 10⁻⁶ s/m²
        β₂ [s²/m]  =  −λ² / (2πc) × D_SI
        β₂ [s²/km] =  β₂ [s²/m] × 10³

    For D=17 ps/(nm·km), λ=1550 nm → β₂ ≈ −21.7 × 10⁻²⁴ s²/km.
    """
    lam_m   = wavelength_nm * 1e-9          # m
    D_SI    = D_ps_nm_km   * 1e-6           # s/m²
    beta2_m = -(lam_m ** 2) * D_SI / (2.0 * np.pi * _C_LIGHT)   # s²/m
    return beta2_m * 1e3                    # s²/km


# ─── SSFM (symmetric split-step) ─────────────────────────────────────────────

def ssfm_propagate(A: np.ndarray,
                   span_km: float,
                   fiber_params: dict,
                   fs: float,
                   dz_km: float = 0.5) -> np.ndarray:
    """
    Propagate the complex optical field A through one fibre span using
    the symmetric Split-Step Fourier Method.

    Symmetric SSFM sequence per step Δz:
        ½ Linear → Full Nonlinear → ½ Linear → … (combined at boundaries)

    Effectively applied as:
        [D/2] NL [D] NL [D] … NL [D/2]

    Parameters
    ----------
    A          : np.ndarray, complex – input field [√W], length N
    span_km    : float  – fibre span length [km]
    fiber_params : dict – see DEFAULT_FIBER
    fs         : float  – sampling frequency [Hz]
    dz_km      : float  – SSFM step size [km]  (smaller → more accurate)

    Returns
    -------
    A_out : np.ndarray, complex – output field after the span [√W]
    """
    # ── Extract and convert parameters ────────────────────────────────────────
    alpha  = _alpha_power_km(fiber_params['alpha_dB_km'])       # neper/km (power)
    beta2  = _beta2_s2_km(fiber_params['D_ps_nm_km'],
                          fiber_params['wavelength_nm'])          # s²/km
    gamma  = fiber_params['gamma_W_km']                          # W⁻¹ km⁻¹

    N      = len(A)
    dt     = 1.0 / fs                                            # s
    omega  = 2.0 * np.pi * np.fft.fftfreq(N, d=dt)              # rad/s

    # Adjust step count so steps divide span evenly
    n_steps = max(1, int(np.ceil(span_km / dz_km)))
    dz      = span_km / n_steps                                  # actual step [km]

    # ── Linear (dispersive + attenuative) transfer functions ──────────────────
    # In the frequency domain: L̂(ω) = −α/2 − j(β₂/2)ω²
    # Over half-step dz/2:  H_half = exp(L̂ · dz/2)
    exponent = (-alpha / 2.0 - 1j * beta2 / 2.0 * omega ** 2)
    H_half   = np.exp(exponent * (dz / 2.0))
    H_full   = H_half ** 2              # full-step = two half-steps

    A_out = A.copy().astype(complex)

    # ── Propagation loop ──────────────────────────────────────────────────────
    # First half-linear step
    A_out = np.fft.ifft(np.fft.fft(A_out) * H_half)

    for step in range(n_steps):
        # Full nonlinear phase rotation (time domain)
        # NL(Δz): A ← A · exp(jγ|A|²Δz)
        A_out *= np.exp(1j * gamma * np.abs(A_out) ** 2 * dz)

        if step < n_steps - 1:
            # Full linear step for interior boundaries (two half-steps merged)
            A_out = np.fft.ifft(np.fft.fft(A_out) * H_full)

    # Final half-linear step
    A_out = np.fft.ifft(np.fft.fft(A_out) * H_half)

    return A_out


# ─── EDFA ─────────────────────────────────────────────────────────────────────

def edfa_amplify(A: np.ndarray,
                 gain_dB: float,
                 edfa_params: dict,
                 fs: float,
                 rng: np.random.Generator = None) -> np.ndarray:
    """
    Amplify the optical field and add ASE (amplified spontaneous emission) noise.

    ASE noise model (single polarisation):
        n_sp  = (G · NF − 1) / (2(G − 1))   – spontaneous emission factor
        S_ASE = (G − 1) · n_sp · h · ν       [W/Hz]  – noise PSD

    For a discrete simulation with sampling rate fs, the noise per sample
    (per quadrature) has standard deviation:
        σ = sqrt(S_ASE · fs / 2)

    Parameters
    ----------
    A          : np.ndarray, complex – input field [√W]
    gain_dB    : float  – power gain [dB]
    edfa_params: dict   – see DEFAULT_EDFA
    fs         : float  – sampling frequency [Hz]
    rng        : np.random.Generator – RNG for reproducibility

    Returns
    -------
    A_out : np.ndarray, complex – amplified field with ASE noise [√W]
    """
    if rng is None:
        rng = np.random.default_rng()

    G  = 10.0 ** (gain_dB / 10.0)                                 # linear power gain
    NF = 10.0 ** (edfa_params['noise_figure_dB'] / 10.0)          # linear noise figure

    # Spontaneous emission factor
    n_sp = (G * NF - 1.0) / (2.0 * (G - 1.0)) if G > 1.0 else NF / 2.0

    # Optical frequency ν [Hz] (assumed fixed at 1550 nm carrier)
    nu    = _C_LIGHT / 1550e-9
    S_ASE = (G - 1.0) * n_sp * _H_PLANCK * nu          # W/Hz

    # Noise std per quadrature per sample
    sigma = np.sqrt(max(S_ASE * fs / 2.0, 0.0))

    # Amplify field (field amplitude scales as √G for power gain G)
    A_out = A * np.sqrt(G)

    # Add complex AWGN
    noise = sigma * (rng.standard_normal(len(A))
                     + 1j * rng.standard_normal(len(A)))
    return A_out + noise


# ─── Full multi-span link ─────────────────────────────────────────────────────

def propagate_link(A: np.ndarray,
                   n_spans: int,
                   span_km: float,
                   fiber_params: dict,
                   edfa_params: dict,
                   fs: float,
                   dz_km: float = 0.5,
                   seed: int = 0) -> np.ndarray:
    """
    Propagate through N identical fibre spans, each followed by an EDFA.

    Architecture:
        [Fiber span 1 → EDFA 1] → [Fiber span 2 → EDFA 2] → … → [Fiber span N → EDFA N]

    The EDFA gain is set to exactly compensate each span's loss.

    Parameters
    ----------
    A          : np.ndarray, complex – launched field [√W]
    n_spans    : int   – number of fibre spans
    span_km    : float – length of each span [km]
    fiber_params : dict – see DEFAULT_FIBER
    edfa_params  : dict – see DEFAULT_EDFA
    fs         : float  – sampling frequency [Hz]
    dz_km      : float  – SSFM step size [km]
    seed       : int    – base random seed; span i uses seed + i

    Returns
    -------
    A_out : np.ndarray, complex – received field [√W]
    """
    # EDFA gain set to exactly compensate the span loss
    gain_dB = fiber_params['alpha_dB_km'] * span_km    # e.g., 0.2 × 80 = 16 dB

    A_out = A.copy()
    for span_idx in range(n_spans):
        A_out = ssfm_propagate(A_out, span_km, fiber_params, fs, dz_km)
        rng   = np.random.default_rng(seed + span_idx)
        A_out = edfa_amplify(A_out, gain_dB, edfa_params, fs, rng)

    return A_out
