# AI-Based Impairment Compensation for High-Speed Optical Communication Links

> **B.Tech (ECE) Final-Year Simulation Project** | Optical Communication + AI/ML (Photonics)

---

## Overview

This project simulates a high-speed single-channel coherent fiber-optic link and demonstrates how a lightweight neural network equalizer can compensate chromatic dispersion and Kerr nonlinearity to improve the **bit error rate (BER)**.

### Key Demonstrations

- Eye diagrams and constellation diagrams **before** and **after** ML equalization
- **BER vs transmission distance** with and without the ML equalizer (80–1600 km)
- Live interactive dashboard with sliders for fiber length and launch power
- Neural network training convergence curve

---

## Physics Model

The complex optical field envelope **A(z, t)** propagates through single-mode fiber according to the **Nonlinear Schrödinger Equation (NLSE)**:

```
∂A/∂z = −(α/2)A − j(β₂/2) ∂²A/∂t² + jγ|A|²A
```

| Symbol | Quantity              | Value used   |
| ------ | --------------------- | ------------ |
| α      | Power attenuation     | 0.2 dB/km    |
| β₂     | Group-velocity disp.  | −21.7 ps²/km |
| γ      | Kerr nonlinear coeff. | 1.3 W⁻¹ km⁻¹ |

Solved numerically using the **symmetric Split-Step Fourier Method (SSFM)**.  
After each 80 km span, an EDFA compensates the loss and adds ASE noise (NF = 5 dB).

---

## Installation

```bash
# 1. Clone / download the project
cd "D:\7th sem Project\optical-link-ml"

# 2. Install dependencies (CPU-only, no GPU required)
pip install -r requirements.txt
```

**Python 3.10+ recommended.**  
All packages run on CPU. PyTorch CPU wheel ≈ 200 MB.

---

## Quick Start

### Step 1 — Train the model and pre-compute the BER curve

```bash
python generate_ber_curve.py
```

This takes a few minutes and produces:

```
models/equalizer.pt
results/ber_curve.npy
results/ber_vs_distance.png
results/training_curve.png
results/constellation_before.png
results/constellation_after.png
```

### Step 2 — Launch the interactive dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## File Structure

```
optical-link-ml/
├── tx.py                    Transmitter: bit generation, QPSK/16-QAM mapping,
│                            RRC pulse shaping, launch power scaling
│
├── fiber.py                 SSFM propagation (NLSE solver) + EDFA ASE noise
│
├── rx.py                    Receiver: matched RRC filter, downsampling,
│                            phase correction (4th-power method), demodulation, BER
│
├── ml.py                    NNEqualizer model (PyTorch), dataset builder,
│                            training loop, inference, model save/load
│
├── plots.py                 Eye diagrams, constellation plots,
│                            BER vs distance, training curves (matplotlib)
│
├── dashboard.py             Streamlit interactive dashboard
│
├── generate_ber_curve.py    Offline: train model + BER sweep script
│
├── models/
│   └── equalizer.pt         Trained PyTorch model weights
│
├── results/
│   ├── ber_curve.npy        BER data array (distances, ber_noml, ber_ml)
│   ├── ber_vs_distance.png
│   ├── training_curve.png
│   ├── constellation_before.png
│   └── constellation_after.png
│
├── requirements.txt
└── README.md
```

---

## System Parameters

| Parameter             | Symbol | Default          | Configurable?         |
| --------------------- | ------ | ---------------- | --------------------- |
| Wavelength            | λ      | 1550 nm          | fiber.py              |
| Symbol rate           | Rₛ     | 32 GBd           | tx.py                 |
| Samples per symbol    | SPS    | 8                | tx.py                 |
| RRC roll-off factor   | β      | 0.1              | tx.py                 |
| Modulation format     | —      | QPSK (default)   | Dashboard             |
| Fibre attenuation     | α      | 0.2 dB/km        | fiber.py              |
| Chromatic dispersion  | D      | 17 ps/(nm·km)    | fiber.py              |
| Nonlinear coefficient | γ      | 1.3 W⁻¹ km⁻¹     | fiber.py              |
| Span length           | —      | 80 km            | fiber.py              |
| EDFA noise figure     | NF     | 5 dB             | fiber.py              |
| Launch power          | P₀     | 0 dBm            | Dashboard slider      |
| SSFM step size        | Δz     | 0.5 km (offline) | generate_ber_curve.py |
| NN context window     | W      | 7 symbols        | ml.py                 |
| NN hidden layer width | —      | 64 neurons       | ml.py                 |
| Training epochs       | —      | 100              | ml.py                 |
| Batch size / lr       | —      | 256 / 1e-3       | ml.py                 |
| Training distance     | —      | 400 km           | generate_ber_curve.py |
| Symbols (training)    | —      | 8192             | generate_ber_curve.py |
| Symbols (live dash)   | —      | 512              | dashboard.py          |

---

## Neural Network Equalizer

**Motivation:** Chromatic dispersion broadens each transmitted pulse into neighbouring symbol periods, causing ISI. Kerr nonlinearity adds power-dependent phase noise. A sliding-window fully-connected network learns to reverse these effects statistically.

**Architecture:**

```
Input: [Re(rx[i−3]), …, Re(rx[i+3]), Im(rx[i−3]), …, Im(rx[i+3])]
                    ↓  FC(64) → ReLU  ↓
                    ↓  FC(64) → ReLU  ↓
           Output: [Re_est(tx[i]), Im_est(tx[i])]
```

- **Loss:** MSE on (Re, Im) — regression output (not classification)
- **Optimiser:** Adam, lr = 1e-3
- **Train/test:** different random seeds → independent test data (no overfitting artefacts)

---

## Expected Results

| Plot                             | What to look for                              |
| -------------------------------- | --------------------------------------------- |
| Eye diagram (back-to-back)       | Wide-open eye                                 |
| Eye diagram (800 km, no ML)      | Closed / distorted eye                        |
| Constellation (QPSK, short link) | Tight clusters at ±45°                        |
| Constellation (long link, ML on) | Clusters tighten vs ML off                    |
| BER vs distance                  | ML curve below no-ML; both rise with distance |
| Training curve                   | Loss decreases and plateaus; no divergence    |

---

## References

[1] E. Ip & J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," _J. Lightw. Technol._, vol. 26, no. 20, pp. 3416–3425, 2008.

[2] M. A. Jarajreh _et al._, "Artificial neural network nonlinear equalizer for coherent optical OFDM," _IEEE Photon. Technol. Lett._, vol. 27, no. 4, pp. 387–390, 2015.

[3] P. J. Freire _et al._, "Neural networks-based equalizers for coherent optical transmission: Caveats and pitfalls," _IEEE J. Sel. Topics Quantum Electron._, vol. 28, no. 4, 2022.

[4] J. Ding _et al._, "Intra-channel nonlinearity mitigation using perturbation-based neural network," _J. Lightw. Technol._, vol. 40, no. 21, 2022.

[5] V. Neskorniuk _et al._, "Neural-network-based nonlinearity equalizer for 128 GBaud coherent transceivers," _Proc. OFC_, 2021.

#
