"""
dashboard.py — Streamlit Interactive Dashboard
================================================
Launch:   streamlit run dashboard.py

Layout
------
Sidebar   : fiber length, launch power, modulation, ML toggle
Tab 1     : Live simulation  → constellation + eye diagram + BER metrics
Tab 2     : Pre-computed     → BER vs distance curve + training curve

Live simulation uses N_SYMBOLS_LIVE = 512 symbols to stay responsive.
Pre-computed curves are loaded from results/ (run generate_ber_curve.py first).
"""
import os, sys
import numpy as np
import streamlit as st

# ── Ensure the package directory is on the path ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tx    import (generate_bits, modulate, pulse_shape,
                   set_launch_power, SYMBOL_RATE_BD, SPS)
from fiber import propagate_link, DEFAULT_FIBER, DEFAULT_EDFA
from rx    import receive, demodulate, compute_ber, phase_correct, matched_filter
from ml    import load_model, equalize, WINDOW
from plots import (plot_constellation, plot_eye_diagram,
                   plot_ber_vs_distance, plot_training_curve)

# ─── Constants ────────────────────────────────────────────────────────────────
N_SYMBOLS_LIVE = 512    # keep the live run fast (< 3 s on a laptop)
SPAN_KM        = 80     # EDFA spacing [km]
DZ_KM_LIVE     = 2.0    # SSFM step for dashboard (coarser, faster)

FS = SPS * SYMBOL_RATE_BD    # sampling rate [Hz]

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'Optical Link AI Dashboard',
    page_icon  = '🔭',
    layout     = 'wide',
)

# ─── Cached resources (loaded once per session) ───────────────────────────────
@st.cache_resource
def _get_model(modulation: str = 'QPSK'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod_slug = '16qam' if modulation == '16-QAM' else 'qpsk'
    path_specific = os.path.join(base_dir, 'models', f'equalizer_{mod_slug}.pt')
    path_default  = os.path.join(base_dir, 'models', 'equalizer.pt')

    if os.path.exists(path_specific):
        return load_model(path_specific)
    elif modulation == 'QPSK' and os.path.exists(path_default):
        return load_model(path_default)
    return None


@st.cache_data
def _load_ber_curve():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'ber_curve.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


@st.cache_data
def _load_training_history():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'training_history.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


# ─── Header ───────────────────────────────────────────────────────────────────
st.title('🔭  Optical Link AI Equalizer Dashboard')
st.caption(
    'AI-Based Impairment Compensation for High-Speed Optical Communication Links  '
    '| B.Tech ECE Final-Year Project'
)

# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Link Parameters')

    fiber_km   = st.slider('Fiber Length (km)',  80,  1600, 400, step=80)
    power_dBm  = st.slider('Launch Power (dBm)', -5,    5,   0, step=1)
    modulation = st.selectbox('Modulation', ['QPSK', '16-QAM'])
    use_ml     = st.toggle('🧠 ML Equalizer', value=True)

    n_spans = fiber_km // SPAN_KM
    st.markdown('---')
    st.markdown(
        f'**Spans:** {n_spans}  \n'
        f'**Total fibre loss:** {0.2 * fiber_km:.0f} dB  \n'
        f'**Total distance:** {fiber_km} km'
    )

    if _get_model(modulation) is None:
        st.warning(f'No trained {modulation} model found.')

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_live, tab_precomp = st.tabs(['📡 Live Results', '📊 Pre-computed Curves'])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    with st.spinner('🔄 Running fibre simulation…'):

        # ── Transmitter ────────────────────────────────────────────────────────
        tx_bits    = generate_bits(N_SYMBOLS_LIVE, modulation, seed=42)
        tx_symbols = modulate(tx_bits, modulation)
        tx_signal  = set_launch_power(pulse_shape(tx_symbols), power_dBm)

        # ── Fibre propagation ──────────────────────────────────────────────────
        if n_spans > 0:
            rx_signal = propagate_link(
                tx_signal, n_spans, SPAN_KM,
                DEFAULT_FIBER, DEFAULT_EDFA,
                fs=FS, dz_km=DZ_KM_LIVE, seed=100,
            )
        else:
            rx_signal = tx_signal.copy()

        # ── Receiver (no equalizer) ────────────────────────────────────────────
        rx_result  = receive(rx_signal, tx_symbols, tx_bits, modulation,
                             total_length_km=fiber_km, fs=FS)
        rx_symbols = rx_result['rx_symbols']
        mf_signal  = rx_result['mf_signal']
        ber_noml   = rx_result['ber']

        # ── ML equalization ────────────────────────────────────────────────────
        eq_symbols = None
        ber_ml     = None
        model      = _get_model(modulation)

        if use_ml and model is not None:
            bps  = 2 if modulation == 'QPSK' else 4
            half = WINDOW // 2
            n    = min(len(rx_symbols), len(tx_symbols))

            eq_raw = equalize(model, rx_symbols[:n])          # equalized IQ (complex)
            n_eq   = len(eq_raw)

            tx_sym_eq  = tx_symbols[half: half + n_eq]
            tx_bits_eq = tx_bits[half * bps: (half + n_eq) * bps]

            eq_pc      = phase_correct(eq_raw, modulation, tx_symbols=tx_sym_eq)
            eq_bits    = demodulate(eq_pc, modulation)
            n_trim     = min(len(tx_bits_eq), len(eq_bits))
            ber_ml     = compute_ber(tx_bits_eq[:n_trim], eq_bits[:n_trim])
            eq_symbols = eq_pc

    # ── Metric row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Distance',      f'{fiber_km} km')
    c2.metric('Launch Power',  f'{power_dBm} dBm')
    c3.metric('BER  (no ML)',  f'{ber_noml:.3e}')

    if ber_ml is not None:
        improvement = (ber_noml - ber_ml) / max(ber_noml, 1e-12) * 100.0
        delta_str   = f'{improvement:+.0f}% improvement'
        c4.metric('BER  (ML on)', f'{ber_ml:.3e}',
                  delta=delta_str, delta_color='normal')
    elif use_ml and model is None:
        c4.metric('BER  (ML on)', 'No model', delta='Run generate_ber_curve.py')
    else:
        c4.metric('BER  (ML on)', '—')

    st.divider()

    # ── Plot row ───────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader('Constellation Diagram')
        syms_list = [tx_symbols[:500], rx_symbols[:500]]
        sym_lbl   = ['Transmitted (ideal)', 'Received (degraded)']
        if eq_symbols is not None:
            syms_list.append(eq_symbols[:500])
            sym_lbl.append('After ML equalizer')

        fig_const = plot_constellation(
            syms_list, sym_lbl,
            title=f'{modulation} — {fiber_km} km — {power_dBm} dBm'
        )
        st.pyplot(fig_const, width='stretch')

    with col_right:
        st.subheader('Eye Diagram  (after matched filter)')
        fig_eye = plot_eye_diagram(
            mf_signal, SPS,
            title=f'{modulation} — {fiber_km} km'
        )
        st.pyplot(fig_eye, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRE-COMPUTED CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_precomp:
    ber_data = _load_ber_curve()
    tr_hist  = _load_training_history()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader('BER vs Transmission Distance')
        if ber_data is not None:
            fig_ber = plot_ber_vs_distance(
                ber_data['distances'],
                ber_data['ber_noml'],
                ber_data['ber_ml'],
            )
            st.pyplot(fig_ber, width='stretch')
        else:
            st.info(
                '📂 No pre-computed data found.  \n'
                'Run `python generate_ber_curve.py` to generate this curve.'
            )

    with col_r:
        st.subheader('Neural Network Training Curve')
        if tr_hist is not None:
            fig_tr = plot_training_curve(tr_hist)
            st.pyplot(fig_tr, width='stretch')
        else:
            st.info(
                '📂 No training history found.  \n'
                'Run `python generate_ber_curve.py` to train the model.'
            )

    if ber_data is not None:
        with st.expander('📋 Raw BER data'):
            import pandas as pd
            df = pd.DataFrame({
                'Distance (km)': ber_data['distances'],
                'BER (no ML)'  : [f'{v:.3e}' for v in ber_data['ber_noml']],
                'BER (ML)'     : [f'{v:.3e}' for v in ber_data['ber_ml']],
            })
            st.dataframe(df, width='stretch')
