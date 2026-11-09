# AI-Based Impairment Compensation for High-Speed Optical Communication Links

> **B.Tech (ECE) Final-Year Simulation Project** | Optical Communication + AI/ML (Photonics)

---

## Overview

This project simulates a high-speed single-channel coherent fiber-optic link (32 GBd) and demonstrates how a lightweight neural network equalizer compensates chromatic dispersion, fiber Kerr nonlinearity, and amplified spontaneous emission (ASE) noise to improve the **bit error rate (BER)**.

### Key Demonstrations

- **Full Coherent DSP Pipeline:** Frequency-domain Chromatic Dispersion Compensation (CDC), Root-Raised-Cosine (RRC) matched filtering, RMS power normalization, and pilot-aided carrier phase recovery (resolving 4-fold quadrant ambiguity).
- **Dual Modulation Support:** Dedicated, trained neural network equalizers for both **QPSK** (2 bits/sym) and **16-QAM** (4 bits/sym).
- **Eye & Constellation Diagrams:** Real-time visual tracking of signal degradation and recovery across fiber distances (80–1600 km).
- **BER vs. Transmission Distance:** Clean, publication-grade log-scale BER curves demonstrating significant BER reduction (e.g. 50% improvement at 1040 km, keeping signals below the $10^{-3}$ FEC threshold).
- **Interactive Streamlit Dashboard:** Live controls for fiber length (80–1600 km), launch power (−5 to +5 dBm), modulation format, and ML equalizer toggle.

---

## Physics & DSP Model

### 1. Optical Fiber Propagation (NLSE)

The complex optical field envelope $A(z, t)$ propagates through single-mode fiber according to the **Nonlinear Schrödinger Equation (NLSE)**:

$$\frac{\partial A}{\partial z} = -\frac{\alpha}{2}A - j\frac{\beta_2}{2} \frac{\partial^2 A}{\partial t^2} + j\gamma|A|^2 A$$

| Symbol            | Quantity                   | Value Used                        | Description                                                            |
| :---------------- | :------------------------- | :-------------------------------- | :--------------------------------------------------------------------- |
| $\alpha$          | Fiber Attenuation          | $0.2\text{ dB/km}$                | Power loss coefficient                                                 |
| $\beta_2$         | Group-Velocity Dispersion  | $-21.67\text{ ps}^2\text{/km}$    | Derived from $D = 17\text{ ps/(nm}\cdot\text{km})$ at $1550\text{ nm}$ |
| $\gamma$          | Kerr Nonlinear Coefficient | $1.3\text{ W}^{-1}\text{km}^{-1}$ | Self-Phase Modulation (SPM)                                            |
| $L_{\text{span}}$ | Span Length                | $80\text{ km}$                    | Periodic amplification spacing                                         |
| $NF$              | EDFA Noise Figure          | $5.0\text{ dB}$                   | Amplified Spontaneous Emission (ASE) noise                             |

The NLSE is solved numerically using the **symmetric Split-Step Fourier Method (SSFM)**. After each 80 km span, an EDFA compensates the span loss and injects ASE noise with noise bandwidth matched to the signal baud rate.

### 2. Receiver Digital Signal Processing (DSP) Pipeline

To recover the transmitted symbols accurately, the coherent receiver follows a structured DSP pipeline:

1. **Chromatic Dispersion Compensation (CDC):** Inverts the accumulated linear chromatic dispersion in the frequency domain:
   $$H_{\text{CDC}}(\omega) = \exp\left(+j \frac{\beta_2}{2} \omega^2 L_{\text{total}}\right)$$
2. **Matched Filtering:** RRC filter matched to the transmitter pulse shaping ($\beta = 0.1$, $\text{span} = 10\text{ symbols}$).
3. **Downsampling & RMS Normalization:** Optimal symbol sampling at 1 sample/symbol, normalized to unit average power.
4. **Carrier Phase Recovery (CPR):** Pilot-aided correlation to eliminate the continuous nonlinear phase shift and resolve 4-fold quadrant symmetry:
   $$\theta = \arg\left( \sum_{k=1}^{N_{\text{pilot}}} r_k \cdot s_k^* \right), \quad r_{\text{corrected}} = r \cdot e^{-j\theta}$$
5. **Neural Network Equalization:** Sliding-window multi-layer perceptron (MLP) compensating residual non-linear phase noise and inter-symbol interference (ISI).
6. **Hard-Decision Demapping & BER Computation:** Minimum Euclidean distance mapping to Gray-coded bit sequences.

---

## Installation

```bash
# 1. Clone or navigate to the project directory
cd "D:\7th sem Project\optical-link-ml"

# 2. Activate virtual environment (if using one)
.venv\Scripts\activate

# 3. Install dependencies (CPU-only, no GPU required)
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, SciPy, PyTorch, Matplotlib, Streamlit, Pandas.

---

## Quick Start

### Step 1 — Run the Complete Test Suite

Verify that all transmitter, fiber, receiver, neural network, and plotting components pass unit checks:

```bash
python test_project.py
```

### Step 2 — Pre-compute the BER Curves & Train Models

Train the equalizer and generate the BER vs. Distance benchmark curve:

```bash
python generate_ber_curve.py
```

This produces:

- `models/equalizer_qpsk.pt` & `models/equalizer.pt`
- `results/ber_curve.npy`
- `results/ber_vs_distance.png`
- `results/training_curve.png`
- `results/constellation_before.png`
- `results/constellation_after.png`

### Step 3 — Launch the Interactive Dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## File Structure

```
optical-link-ml/
├── tx.py                  # Transmitter: PRBS bit generation, QPSK/16-QAM mapping,
│                          # RRC pulse shaping, optical launch power scaling
│
├── fiber.py               # SSFM propagation (NLSE solver) + EDFA ASE noise model
│
├── rx.py                  # Receiver: CDC equalizer, matched RRC filter, downsampler,
│                          # RMS power normalizer, pilot carrier phase recovery, BER
│
├── ml.py                  # PyTorch NNEqualizer (W=7 window, 5250 params), dataset builder,
│                          # training loop, inference equalization, model I/O
│
├── plots.py               # Visualizations: eye diagrams, constellation scatter (with
│                          # layered z-order), log-scale BER curves, loss curves
│
├── dashboard.py           # Streamlit app with dynamic model loading (QPSK & 16-QAM)
│                          # and modern container width layout
│
├── generate_ber_curve.py  # Offline script: model training + BER distance sweep
│
├── test_project.py        # Automated test suite covering the full transceiver chain
│
├── models/
│   ├── equalizer.pt       # Default active model checkpoint
│   ├── equalizer_qpsk.pt  # Dedicated QPSK trained equalizer
│   └── equalizer_16qam.pt # Dedicated 16-QAM trained equalizer
│
├── results/
│   ├── ber_curve.npy      # Cached BER data array (distances, ber_noml, ber_ml)
│   ├── ber_vs_distance.png
│   ├── training_curve.png
│   ├── constellation_before.png
│   └── constellation_after.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## System Parameters

| Parameter             | Symbol            | Default Value                                    | Configurable Location                    |
| :-------------------- | :---------------- | :----------------------------------------------- | :--------------------------------------- |
| Optical Wavelength    | $\lambda$         | $1550\text{ nm}$                                 | `fiber.py`                               |
| Symbol Rate           | $R_s$             | $32\text{ GBd}$                                  | `tx.py`                                  |
| Samples per Symbol    | SPS               | $8$ ($F_s = 256\text{ GHz}$)                     | `tx.py`                                  |
| RRC Roll-off Factor   | $\beta$           | $0.1$                                            | `tx.py`                                  |
| RRC Filter Span       | —                 | $10\text{ symbols}$                              | `tx.py`                                  |
| Modulation Formats    | —                 | QPSK (default), 16-QAM                           | Dashboard / `tx.py`                      |
| Fiber Attenuation     | $\alpha$          | $0.2\text{ dB/km}$                               | `fiber.py`                               |
| Chromatic Dispersion  | $D$               | $17\text{ ps/(nm}\cdot\text{km})$                | `fiber.py`                               |
| Nonlinear Kerr Coeff. | $\gamma$          | $1.3\text{ W}^{-1}\text{km}^{-1}$                | `fiber.py`                               |
| Span Length           | $L_{\text{span}}$ | $80\text{ km}$                                   | `fiber.py`                               |
| EDFA Noise Figure     | NF                | $5.0\text{ dB}$                                  | `fiber.py`                               |
| Launch Power          | $P_0$             | $0\text{ dBm}$ (live), $+3.5\text{ dBm}$ (sweep) | Dashboard / `generate_ber_curve.py`      |
| SSFM Step Size        | $\Delta z$        | $2.0\text{ km}$                                  | `dashboard.py` / `generate_ber_curve.py` |
| NN Context Window     | $W$               | $7\text{ symbols}$ ($2W = 14$ inputs)            | `ml.py`                                  |
| NN Hidden Neurons     | —                 | $64 \times 64$ ($5,250$ weights)                 | `ml.py`                                  |
| Training Epochs / LR  | —                 | $100\text{ epochs}$, Adam $\text{lr}=10^{-3}$    | `ml.py`                                  |
| Training Distance     | —                 | $400\text{ km}$ ($5\text{ spans}$)               | `generate_ber_curve.py`                  |
| Symbols (Training)    | —                 | $8,192\text{ symbols}$                           | `generate_ber_curve.py`                  |
| Symbols (Live Dash)   | —                 | $512\text{ symbols}$                             | `dashboard.py`                           |

---

## Neural Network Equalizer Architecture

**Motivation:** After frequency-domain CDC removes bulk chromatic dispersion, residual dispersion, intra-channel cross-phase modulation, and self-phase modulation (SPM) cause signal cloud deformation and inter-symbol interference. A sliding-window neural network reconstructs the undistorted transmitted symbols.

```
Input: [Re(rx[i-3...i+3]), Im(rx[i-3...i+3])]  (14 inputs)
                       ↓
            Linear(14 → 64) + ReLU
                       ↓
            Linear(64 → 64) + ReLU
                       ↓
             Linear(64 → 2)
                       ↓
      Output: [Re_est(tx[i]), Im_est(tx[i])]
```

- **Objective:** Mean Squared Error (MSE) loss on IQ coordinates (Regression).
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$).
- **Window Size:** $W = 7$ symbols (captures $\pm 3$ neighboring symbol ISI context).
- **Separation:** Training and test datasets generated with distinct pseudo-random seeds.

---

## Expected Results & Verification

| Metric / Diagram               | Expected Behavior                                                                                                                                                   |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Eye Diagram (Back-to-Back)** | Wide-open eye with zero inter-symbol interference.                                                                                                                  |
| **Eye Diagram (Impaired)**     | Eye openings narrow with distance and power due to noise and ISI.                                                                                                   |
| **Constellation Diagram**      | Ideal transmitted points (blue) layered cleanly on top of received (red) and equalized (green) clusters.                                                            |
| **QPSK Equalization**          | Tightening of the 4 quadrant clouds around $(\pm 0.707, \pm 0.707)$.                                                                                                |
| **16-QAM Equalization**        | Resolution of the $4 \times 4$ grid using the dedicated 16-QAM model.                                                                                               |
| **BER vs. Distance Curve**     | Error-free at short-to-mid reach ($< 880\text{ km}$), rising monotonically at longer reaches, with the ML equalizer consistently outperforming the non-ML baseline. |
| **Training Loss Curve**        | Loss steadily decreases from $\approx 0.4$ down to $\le 10^{-4}$ without divergence.                                                                                |

---

## References

1. E. Ip & J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," _J. Lightw. Technol._, vol. 26, no. 20, pp. 3416–3425, 2008.
2. M. A. Jarajreh _et al._, "Artificial neural network nonlinear equalizer for coherent optical OFDM," _IEEE Photon. Technol. Lett._, vol. 27, no. 4, pp. 387–390, 2015.
3. P. J. Freire _et al._, "Neural networks-based equalizers for coherent optical transmission: Caveats and pitfalls," _IEEE J. Sel. Topics Quantum Electron._, vol. 28, no. 4, 2022.
4. J. Ding _et al._, "Intra-channel nonlinearity mitigation using perturbation-based neural network," _J. Lightw. Technol._, vol. 40, no. 21, 2022.
5. V. Neskorniuk _et al._, "Neural-network-based nonlinearity equalizer for 128 GBaud coherent transceivers," _Proc. OFC_, 2021.
