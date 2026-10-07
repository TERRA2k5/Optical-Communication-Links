import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np

print('--- tx.py ---')
from tx import generate_bits, modulate, pulse_shape, set_launch_power, SPS, SYMBOL_RATE_BD
bits = generate_bits(512, 'QPSK', seed=42)
syms = modulate(bits, 'QPSK')
sig  = set_launch_power(pulse_shape(syms), 0.0)
pwr  = 10*np.log10(np.mean(np.abs(sig)**2)*1000)
print('  Launch power:', round(pwr,3), 'dBm  [expect 0.000]')
assert abs(pwr) < 0.01

print()
print('--- fiber.py ---')
from fiber import propagate_link, DEFAULT_FIBER, DEFAULT_EDFA, _beta2_s2_km
b2 = _beta2_s2_km(17.0, 1550.0)
print('  beta2:', '{:.3e}'.format(b2), 's^2/km  [expect -2.167e-23]')
fs = SPS * SYMBOL_RATE_BD
rx1 = propagate_link(sig, 1, 80, DEFAULT_FIBER, DEFAULT_EDFA, fs=fs, dz_km=2.0, seed=0)
pwr2 = 10*np.log10(np.mean(np.abs(rx1)**2)*1000)
print('  Power after 1 span + EDFA:', round(pwr2,2), 'dBm  [expect ~0]')
assert abs(pwr2) < 5

print()
print('--- rx.py: back-to-back BER ---')
from rx import receive
bits2 = generate_bits(1024, 'QPSK', seed=42)
syms2 = modulate(bits2, 'QPSK')
sig2  = set_launch_power(pulse_shape(syms2), 0.0)
res   = receive(sig2, syms2, bits2, 'QPSK')
print('  Back-to-back BER:', res['ber'], '  [expect 0.0]')
assert res['ber'] == 0.0, 'FAIL: back-to-back BER != 0 => ' + str(res['ber'])

print()
print('--- rx.py: fibre-impaired BER > 0 ---')
sig3 = set_launch_power(pulse_shape(syms2), 0.0)
rx3  = propagate_link(sig3, 5, 80, DEFAULT_FIBER, DEFAULT_EDFA, fs=fs, dz_km=2.0, seed=7)
res3 = receive(rx3, syms2, bits2, 'QPSK')
print('  BER after 400 km:', round(res3['ber'], 4), '  [expect > 0]')
assert res3['ber'] > 0

print()
print('--- ml.py ---')
from ml import NNEqualizer, build_dataset, train, equalize
rx_sym = res3['rx_symbols']
tx_sym = syms2[:len(rx_sym)]
X, Y = build_dataset(rx_sym, tx_sym, window=7)
print('  Dataset X:', X.shape, '  Y:', Y.shape)
model = NNEqualizer()
print('  Model params:', sum(p.numel() for p in model.parameters()), '[expect 5250]')
hist = train(model, X, Y, epochs=5, batch_size=128)
print('  Final train loss:', round(hist['train_loss'][-1], 6))
eq_sym = equalize(model, rx_sym)
print('  Equalized symbols:', len(eq_sym))

print()
print('--- plots.py ---')
from plots import plot_eye_diagram, plot_constellation, plot_ber_vs_distance, plot_training_curve
fig1 = plot_eye_diagram(sig2[:2000], SPS, 'Eye Test')
fig2 = plot_constellation([syms2[:300], rx_sym[:300]], ['TX','RX'], 'Const Test')
fig3 = plot_ber_vs_distance(np.array([80.,160.,240.]), np.array([0.1,0.2,0.4]), np.array([0.05,0.1,0.2]))
fig4 = plot_training_curve(hist)
print('  All 4 figure functions OK')

print()
print('=' * 45)
print('  ALL TESTS PASSED')
print('=' * 45)
