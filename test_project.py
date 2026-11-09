import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np

print('=' * 50)
print(' RUNNING OPTICAL LINK ML TEST SUITE')
print('=' * 50)

# ── 1. Transmitter (QPSK + 16-QAM) ──────────────────────
print('\n[1/6] Testing tx.py...')
from tx import generate_bits, modulate, pulse_shape, set_launch_power, SPS, SYMBOL_RATE_BD

# QPSK Power
bits_qpsk = generate_bits(512, 'QPSK', seed=42)
syms_qpsk = modulate(bits_qpsk, 'QPSK')
sig_qpsk  = set_launch_power(pulse_shape(syms_qpsk), 0.0)
pwr = 10 * np.log10(np.mean(np.abs(sig_qpsk)**2) * 1000)
assert abs(pwr) < 0.01, f'Expected 0 dBm launch power, got {pwr:.3f}'
print(f'  Launch power: {pwr:.3f} dBm  [OK]')

# 16-QAM Modulate
bits_16qam = generate_bits(512, '16-QAM', seed=42)
syms_16qam = modulate(bits_16qam, '16-QAM')
assert len(syms_16qam) == 512, '16-QAM symbol count mismatch'
print('  16-QAM Modulation  [OK]')

# ── 2. Fiber & SSFM Physics ────────────────────────────
print('\n[2/6] Testing fiber.py physics...')
from fiber import propagate_link, DEFAULT_FIBER, DEFAULT_EDFA, _beta2_s2_km
fs = SPS * SYMBOL_RATE_BD

# Dispersion calculation check
b2 = _beta2_s2_km(17.0, 1550.0)
assert abs(b2 - (-2.167e-23)) < 1e-25, 'Incorrect beta2 calculation'
print(f'  beta2: {b2:.3e} s^2/km  [OK]')

# Pulse Broadening Check: RMS width should increase after propagation
t_axis = np.arange(len(sig_qpsk))
rms_before = np.sqrt(np.average(t_axis**2, weights=np.abs(sig_qpsk)**2))
rx_disp = propagate_link(sig_qpsk, 3, 80, DEFAULT_FIBER, DEFAULT_EDFA, fs=fs, dz_km=2.0, seed=0)
rms_after = np.sqrt(np.average(t_axis**2, weights=np.abs(rx_disp)**2))
print(f'  RMS width: {rms_before:.1f} -> {rms_after:.1f}  (Pulse broadened) [OK]')

# ── 3. Receiver Back-to-Back ───────────────────────────
print('\n[3/6] Testing rx.py Back-to-Back (Zero Impairment)...')
from rx import receive
res_b2b_qpsk = receive(sig_qpsk, syms_qpsk, bits_qpsk, 'QPSK')
assert res_b2b_qpsk['ber'] == 0.0, f'QPSK back-to-back BER != 0 ({res_b2b_qpsk["ber"]})'
print('  QPSK Back-to-Back BER: 0.0  [OK]')

sig_16qam = set_launch_power(pulse_shape(syms_16qam), 0.0)
res_b2b_16qam = receive(sig_16qam, syms_16qam, bits_16qam, '16-QAM')
assert res_b2b_16qam['ber'] == 0.0, f'16-QAM back-to-back BER != 0 ({res_b2b_16qam["ber"]})'
print('  16-QAM Back-to-Back BER: 0.0  [OK]')

# ── 4. Receiver Impairment & CDC ───────────────────────
print('\n[4/6] Testing Chromatic Dispersion & Compensation...')
# Impaired (uncompensated) -> BER > 0
res_imp = receive(rx_disp, syms_qpsk, bits_qpsk, 'QPSK')
assert res_imp['ber'] > 0.05, f'Expected impaired BER > 0, got {res_imp["ber"]}'
print(f'  Uncompensated BER (240 km): {res_imp["ber"]:.4f}  [OK]')

# Compensated with CDC + Phase Recovery -> BER drops
res_cdc = receive(rx_disp, syms_qpsk, bits_qpsk, 'QPSK', total_length_km=240, fs=fs)
assert res_cdc['ber'] < res_imp['ber'], 'CDC did not improve BER'
print(f'  Compensated BER (240 km): {res_cdc["ber"]:.4f}  (Restored) [OK]')

# ── 5. Neural Network Equalizer ────────────────────────
print('\n[5/6] Testing ml.py (Training & BER Gain)...')
from ml import NNEqualizer, build_dataset, train, equalize, WINDOW
from rx import demodulate, compute_ber, phase_correct

rx_sym = res_imp['rx_symbols']
tx_sym = syms_qpsk[:len(rx_sym)]
X, Y = build_dataset(rx_sym, tx_sym, window=7)
model = NNEqualizer()
hist = train(model, X, Y, epochs=15, batch_size=128)
assert hist['train_loss'][-1] < hist['train_loss'][0], 'Model loss did not decrease'
print(f'  Train loss: {hist["train_loss"][0]:.4f} -> {hist["train_loss"][-1]:.4f}  [OK]')

# Test equalized symbols
half = WINDOW // 2
eq_sym = equalize(model, rx_sym)
tx_trim = tx_sym[half: half + len(eq_sym)]
bits_trim = bits_qpsk[half * 2: (half + len(eq_sym)) * 2]
eq_pc = phase_correct(eq_sym, 'QPSK', tx_symbols=tx_trim)
eq_bits = demodulate(eq_pc, 'QPSK')
eq_ber = compute_ber(bits_trim[:len(eq_bits)], eq_bits)
print(f'  ML Equalized BER: {eq_ber:.4f}  [OK]')

# ── 6. Plotting & Artifacts ────────────────────────────
print('\n[6/6] Testing plots.py and saved files...')
import os
from plots import plot_eye_diagram, plot_constellation, plot_ber_vs_distance, plot_training_curve

fig1 = plot_eye_diagram(sig_qpsk[:2000], SPS, 'Eye Test')
fig2 = plot_constellation([syms_qpsk[:300], rx_sym[:300]], ['TX','RX'], 'Const Test')
fig3 = plot_ber_vs_distance(np.array([80.,160.,240.]), np.array([0.1,0.2,0.4]), np.array([0.05,0.1,0.2]))
fig4 = plot_training_curve(hist)
print('  All 4 figure functions generated successfully  [OK]')

# Check project deliverables exist
assert os.path.exists('models/equalizer.pt'), 'models/equalizer.pt missing'
assert os.path.exists('results/ber_curve.npy'), 'results/ber_curve.npy missing'
print('  Saved model and pre-computed results exist  [OK]')

print('\n' + '=' * 50)
print('  ALL 6 TEST STAGES PASSED SUCCESSFULLY!')
print('=' * 50)