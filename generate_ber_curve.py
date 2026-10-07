"""
generate_ber_curve.py — Offline Training and BER Sweep Script
==============================================================
Run this script ONCE before launching the dashboard:

    python generate_ber_curve.py

What it does
------------
1. Generates training data at a moderate fibre length (400 km / 5 spans).
2. Trains the NNEqualizer and saves the weights to models/equalizer.pt.
3. Sweeps transmission distance from 80 km to 1600 km, computing BER
   with and without the ML equalizer at each point.
4. Saves BER data to results/ber_curve.npy and generates result figures.

Training / test data use different random seeds (NFR-3: reproducibility,
FR-8: independent test data).

Outputs
-------
    models/equalizer.pt
    results/ber_curve.npy
    results/ber_vs_distance.png
    results/training_curve.png
    results/training_history.npy
    results/constellation_before.png
    results/constellation_after.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tx    import (generate_bits, modulate, pulse_shape,
                   set_launch_power, SYMBOL_RATE_BD, SPS)
from fiber import propagate_link, DEFAULT_FIBER, DEFAULT_EDFA
from rx    import receive, demodulate, compute_ber, phase_correct
from ml    import (NNEqualizer, build_dataset, train as train_model,
                   equalize, save_model, WINDOW)
from plots import (plot_ber_vs_distance, plot_training_curve,
                   plot_constellation, save_figure)

# ─── Configuration ────────────────────────────────────────────────────────────
MODULATION      = 'QPSK'
POWER_DBM       = 0.0
SPAN_KM         = 80         # km per EDFA span
N_TRAIN_SPANS   = 5          # train at 5 × 80 = 400 km
N_SYMBOLS_TRAIN = 8192       # symbols for training
N_SYMBOLS_TEST  = 4096       # symbols per BER test point
DZ_KM           = 0.5        # SSFM step size [km]  (accuracy)
DISTANCES_KM    = list(range(80, 1601, 80))   # 80, 160, …, 1600

BPS = 2 if MODULATION == 'QPSK' else 4
FS  = SPS * SYMBOL_RATE_BD   # sampling frequency [Hz]

os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
print('=' * 65)
print(' AI-Based Optical Link Impairment Compensation')
print(' generate_ber_curve.py')
print('=' * 65)

# ─── Step 1: Generate training data ───────────────────────────────────────────
t0 = time.time()
print(f'\n[1/4] Generating training data at {N_TRAIN_SPANS * SPAN_KM} km '
      f'({N_SYMBOLS_TRAIN} symbols, seed=42)')

tx_bits_tr    = generate_bits(N_SYMBOLS_TRAIN, MODULATION, seed=42)
tx_symbols_tr = modulate(tx_bits_tr, MODULATION)
tx_signal_tr  = set_launch_power(pulse_shape(tx_symbols_tr), POWER_DBM)

rx_signal_tr  = propagate_link(
    tx_signal_tr, N_TRAIN_SPANS, SPAN_KM,
    DEFAULT_FIBER, DEFAULT_EDFA,
    fs=FS, dz_km=DZ_KM, seed=0,
)

rx_result_tr  = receive(rx_signal_tr, tx_symbols_tr, tx_bits_tr, MODULATION)
rx_symbols_tr = rx_result_tr['rx_symbols']
ber_train_noml = rx_result_tr['ber']
print(f'    BER (no ML, training distance): {ber_train_noml:.3e}')

# Align lengths for dataset construction
n_min         = min(len(rx_symbols_tr), len(tx_symbols_tr))
rx_symbols_tr = rx_symbols_tr[:n_min]
tx_symbols_tr = tx_symbols_tr[:n_min]

X_train, Y_train = build_dataset(rx_symbols_tr, tx_symbols_tr, WINDOW)
print(f'    Training samples: {len(X_train)}  '
      f'({time.time() - t0:.1f} s)')

# ─── Step 2: Train the model ───────────────────────────────────────────────────
print(f'\n[2/4] Training NNEqualizer  (epochs=100, batch=256, lr=1e-3)')
t1    = time.time()
model = NNEqualizer()
history = train_model(model, X_train, Y_train)
save_model(model, 'models/equalizer.pt')
np.save('results/training_history.npy', history)

fig_tr = plot_training_curve(history)
save_figure(fig_tr, 'results/training_curve.png')
print(f'    Training complete  ({time.time() - t1:.1f} s)')
print(f'    Final train loss : {history["train_loss"][-1]:.6f}')
print(f'    Final val   loss : {history["val_loss"][-1]:.6f}')

# ─── Step 3: BER vs distance sweep ────────────────────────────────────────────
print(f'\n[3/4] BER vs distance sweep  ({len(DISTANCES_KM)} points)')
print(f'    {"Dist (km)":<12} {"n_spans":<10} {"BER no-ML":<14} {"BER ML":<14}')
print('    ' + '-' * 52)

bers_noml = []
bers_ml   = []

for dist_km in DISTANCES_KM:
    n_spans   = dist_km // SPAN_KM
    seed_test = 1000 + dist_km    # distinct from training seed (42)

    tx_bits_te    = generate_bits(N_SYMBOLS_TEST, MODULATION, seed=seed_test)
    tx_symbols_te = modulate(tx_bits_te, MODULATION)
    tx_signal_te  = set_launch_power(pulse_shape(tx_symbols_te), POWER_DBM)

    rx_signal_te  = propagate_link(
        tx_signal_te, n_spans, SPAN_KM,
        DEFAULT_FIBER, DEFAULT_EDFA,
        fs=FS, dz_km=DZ_KM, seed=seed_test,
    )

    rx_result_te  = receive(rx_signal_te, tx_symbols_te, tx_bits_te, MODULATION)
    rx_sym_te     = rx_result_te['rx_symbols']
    ber_no_ml     = rx_result_te['ber']

    # ML equalization on test data
    half          = WINDOW // 2
    n_te          = min(len(rx_sym_te), len(tx_symbols_te))
    eq_raw        = equalize(model, rx_sym_te[:n_te])
    n_eq          = len(eq_raw)
    tx_sym_trim   = tx_symbols_te[half: half + n_eq]
    tx_bits_trim  = tx_bits_te[half * BPS: (half + n_eq) * BPS]

    eq_pc         = phase_correct(eq_raw, MODULATION)
    eq_bits       = demodulate(eq_pc, MODULATION)
    n_trim        = min(len(tx_bits_trim), len(eq_bits))
    ber_ml_val    = compute_ber(tx_bits_trim[:n_trim], eq_bits[:n_trim])

    bers_noml.append(ber_no_ml)
    bers_ml.append(ber_ml_val)

    print(f'    {dist_km:<12} {n_spans:<10} {ber_no_ml:<14.3e} {ber_ml_val:<14.3e}')

# ─── Step 4: Save results ─────────────────────────────────────────────────────
print(f'\n[4/4] Saving results')

ber_curve = {
    'distances': np.array(DISTANCES_KM),
    'ber_noml' : np.array(bers_noml),
    'ber_ml'   : np.array(bers_ml),
}
np.save('results/ber_curve.npy', ber_curve)

fig_ber = plot_ber_vs_distance(
    ber_curve['distances'],
    ber_curve['ber_noml'],
    ber_curve['ber_ml'],
)
save_figure(fig_ber, 'results/ber_vs_distance.png')

# Constellation snapshots at training distance
fig_before = plot_constellation(
    [tx_symbols_tr[:500], rx_symbols_tr[:500]],
    ['Transmitted', f'Received ({N_TRAIN_SPANS * SPAN_KM} km, no ML)'],
    title=f'Constellation — Before ML Equalization ({N_TRAIN_SPANS * SPAN_KM} km)',
)
save_figure(fig_before, 'results/constellation_before.png')

# Equalized symbols at training distance
eq_tr    = equalize(model, rx_symbols_tr)
half     = WINDOW // 2
eq_tr_pc = phase_correct(eq_tr, MODULATION)
fig_after = plot_constellation(
    [tx_symbols_tr[half: half + len(eq_tr)][:500],
     rx_symbols_tr[half: half + len(eq_tr)][:500],
     eq_tr_pc[:500]],
    ['Transmitted', f'Received ({N_TRAIN_SPANS * SPAN_KM} km)', 'After ML equalizer'],
    title=f'Constellation — After ML Equalization ({N_TRAIN_SPANS * SPAN_KM} km)',
)
save_figure(fig_after, 'results/constellation_after.png')

total_time = time.time() - t0
print(f'\n✅  All done!  Total time: {total_time:.1f} s')
print()
print('  Saved files:')
print('    models/equalizer.pt')
print('    results/ber_curve.npy')
print('    results/ber_vs_distance.png')
print('    results/training_curve.png')
print('    results/training_history.npy')
print('    results/constellation_before.png')
print('    results/constellation_after.png')
print()
print('  Launch dashboard:')
print('    streamlit run dashboard.py')
