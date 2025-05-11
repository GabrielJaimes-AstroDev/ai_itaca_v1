import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from core_functions import *
import tempfile
import plotly.graph_objects as go
import tensorflow as tf
import gdown
import shutil
from scipy.stats import linregress
from scipy.signal import find_peaks

st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# === CUSTOM CSS STYLES ===
st.markdown("""
<style>
    /* (Mantener todos los estilos CSS anteriores igual) */
</style>
""", unsafe_allow_html=True)

# === HEADER WITH IMAGE AND DESCRIPTION ===
st.image("NGC6523_BVO_2.jpg", use_container_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

# Project description (mantener igual)

# === CONFIGURATION ===
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1zlnkEoRvHR1CoK9hXxD0Jy4JIKF5Uybz?usp=sharing"
TEMP_MODEL_DIR = "downloaded_models"

if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

@st.cache_data(show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    # (Mantener esta función igual)

def find_significant_coincident_peaks(input_freq, input_spec, best_match_freq, best_match_spec, sigma_threshold=1.0, tolerance=0.1):
    """Find significant coincident peaks between input and best match spectra"""
    # Calculate baseline statistics
    input_mean = np.mean(input_spec)
    input_std = np.std(input_spec)
    detection_threshold = input_mean + sigma_threshold * input_std
    
    # Find all peaks in input spectrum above sigma threshold
    input_peaks, _ = find_peaks(input_spec, height=detection_threshold)
    input_peak_freqs = input_freq[input_peaks]
    input_peak_ints = input_spec[input_peaks]
    
    # Find all peaks in best match spectrum
    best_match_peaks, _ = find_peaks(best_match_spec)
    best_match_peak_freqs = best_match_freq[best_match_peaks]
    best_match_peak_ints = best_match_spec[best_match_peaks]
    
    # Find coincident peaks within frequency tolerance
    coincident_peaks = []
    for i, input_f in enumerate(input_peak_freqs):
        for j, match_f in enumerate(best_match_peak_freqs):
            if abs(input_f - match_f) <= tolerance:
                coincident_peaks.append({
                    'input_freq': input_f,
                    'input_intensity': input_peak_ints[i],
                    'match_freq': match_f,
                    'match_intensity': best_match_peak_ints[j],
                    'delta_freq': abs(input_f - match_f),
                    'sigma_level': (input_peak_ints[i] - input_mean) / input_std
                })
                break
    
    return coincident_peaks, detection_threshold, input_mean, input_std

def calculate_errors_from_coincident_peaks(coincident_peaks):
    """Calculate errors based on coincident peaks"""
    if not coincident_peaks:
        return None
    
    input_ints = np.array([p['input_intensity'] for p in coincident_peaks])
    match_ints = np.array([p['match_intensity'] for p in coincident_peaks])
    
    # Calculate various error metrics
    abs_errors = np.abs(input_ints - match_ints)
    rel_errors = abs_errors / (match_ints + 1e-10)
    
    slope, intercept, r_value, p_value, std_err = linregress(input_ints, match_ints)
    
    return {
        'mean_abs_error': np.mean(abs_errors),
        'mean_rel_error': np.mean(rel_errors),
        'max_abs_error': np.max(abs_errors),
        'max_rel_error': np.max(rel_errors),
        'correlation': r_value,
        'slope': slope,
        'intercept': intercept,
        'std_error': std_err,
        'num_coincident_peaks': len(coincident_peaks)
    }

def evaluate_molecule_presence(coincident_peaks):
    """Evaluate if molecule is present based on significant coincident peaks"""
    if not coincident_peaks:
        return {
            'present': False,
            'confidence': 'none',
            'message': 'No coincident peaks found between input and best match',
            'num_peaks': 0
        }
    
    # Count significant coincident peaks (>1σ)
    num_sig_peaks = len(coincident_peaks)  # All peaks passed the sigma threshold
    
    if num_sig_peaks == 0:
        return {
            'present': False,
            'confidence': 'none',
            'message': 'No coincident peaks above detection threshold',
            'num_peaks': 0
        }
    elif num_sig_peaks < 3:
        return {
            'present': True,
            'confidence': 'weak',
            'message': f'{num_sig_peaks} coincident peak(s) above detection threshold - weak detection',
            'num_peaks': num_sig_peaks
        }
    else:
        return {
            'present': True,
            'confidence': 'strong',
            'message': f'{num_sig_peaks} coincident peaks above detection threshold - confident detection',
            'num_peaks': num_sig_peaks
        }

# === SIDEBAR ===
st.sidebar.title("Configuration")

model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=None,
    help="Drag and drop file here. Limit 200MB per file"
)

st.sidebar.subheader("Peak Matching Parameters")
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 1.0, step=0.1, 
                                   help="Threshold for significant peak detection (in standard deviations)")
tolerance_ghz = st.sidebar.slider("Tolerance (GHz)", 0.01, 0.5, 0.1, step=0.01,
                                 help="Frequency tolerance for coincident peaks")

# (Mantener el resto de parámetros igual)

# === MAIN APP ===
st.title("Molecular Spectrum Analyzer | AI - ITACA")

if input_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name

    if not model_files:
        st.error("No trained models were found in Google Drive.")
    else:
        selected_model = st.selectbox("Select Molecule Model", model_files)

        if st.button("Analyze Spectrum"):
            try:
                with st.spinner("Analyzing spectrum..."):
                    mol_name = selected_model.replace('_model.keras', '')
                    model_path = os.path.join(TEMP_MODEL_DIR, selected_model)
                    model = tf.keras.models.load_model(model_path)

                    data_file = os.path.join(TEMP_MODEL_DIR, f'{mol_name}_train_data.npz')
                    if not os.path.exists(data_file):
                        st.error(f"Training data not found for {mol_name}")
                    else:
                        with np.load(data_file) as data:
                            train_freq = data['train_freq']
                            train_data = data['train_data']
                            train_logn = data['train_logn']
                            train_tex = data['train_tex']
                            headers = data['headers']
                            filenames = data['filenames']

                        results = analyze_spectrum(
                            tmp_path, model, train_data, train_freq,
                            filenames, headers, train_logn, train_tex,
                            config, mol_name
                        )

                        if results.get('best_match'):
                            # Find significant coincident peaks
                            coincident_peaks, detection_threshold, input_mean, input_std = find_significant_coincident_peaks(
                                results['input_freq'], 
                                results['input_spec'],
                                results['best_match']['x_synth'],
                                results['best_match']['y_synth'],
                                sigma_threshold=sigma_threshold,
                                tolerance=tolerance_ghz
                            )
                            
                            # Evaluate molecule presence
                            presence = evaluate_molecule_presence(coincident_peaks)
                            peak_errors = calculate_errors_from_coincident_peaks(coincident_peaks)
                            
                            # Display detection result
                            if presence['present']:
                                if presence['confidence'] == 'strong':
                                    st.markdown(f"""
                                    <div class="molecule-detected">
                                        <h4>✅ Molecule Detected (Confident)</h4>
                                        <p>{presence['message']}</p>
                                        <p><strong>Detection Threshold:</strong> {sigma_threshold}σ ({detection_threshold:.2f} intensity)</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="weak-detection">
                                        <h4>⚠️ Molecule Detected (Weak)</h4>
                                        <p>{presence['message']}</p>
                                        <p><strong>Detection Threshold:</strong> {sigma_threshold}σ ({detection_threshold:.2f} intensity)</p>
                                        <p>Results should be interpreted with caution</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="no-molecule-detected">
                                    <h4>❌ Molecule Not Detected</h4>
                                    <p>{presence['message']}</p>
                                    <p><strong>Detection Threshold:</strong> {sigma_threshold}σ ({detection_threshold:.2f} intensity)</p>
                                </div>
                                """, unsafe_allow_html=True)

                            st.success("Analysis completed successfully!")

                            tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "Interactive Summary", 
                                "Molecule Best Match", 
                                "Peak Matching", 
                                "CNN Training", 
                                "Top Selection: LogN", 
                                "Top Selection: Tex"
                            ])

                            with tab0:
                                col1, col2 = st.columns(2)
                                with col1:
                                    if peak_errors:
                                        st.markdown(f"""
                                        <div style="
                                            background-color: #f0f8ff;
                                            padding: 15px;
                                            border-radius: 10px;
                                            border-left: 5px solid #1E88E5;
                                            margin-bottom: 20px;
                                        ">
                                            <h4 style="color: #1E88E5; margin-top: 0;">Detection of Physical Parameters</h4>
                                            <p><strong>LogN:</strong> {results['best_match']['logn']:.2f} ± {peak_errors['mean_rel_error']*results['best_match']['logn']:.2f} cm⁻²</p>
                                            <p><strong>Tex:</strong> {results['best_match']['tex']:.2f} ± {peak_errors['mean_rel_error']*results['best_match']['tex']:.2f} K</p>
                                            <p><strong>Coincident Peaks:</strong> {peak_errors['num_coincident_peaks']}</p>
                                            <p><strong>Mean Relative Error:</strong> {peak_errors['mean_rel_error']*100:.1f}%</p>
                                            <p><strong>Correlation:</strong> {peak_errors['correlation']:.3f}</p>
                                            <p><strong>Detection Threshold:</strong> {sigma_threshold}σ ({detection_threshold:.2f})</p>
                                            <p><strong>File (Top CNN Train):</strong> {results['best_match']['filename']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                fig = go.Figure()
                                
                                # Input spectrum
                                fig.add_trace(go.Scatter(
                                    x=results['input_freq'],
                                    y=results['input_spec'],
                                    mode='lines',
                                    name='Input Spectrum',
                                    line=dict(color='white', width=2))
                                )
                                
                                # Best match spectrum
                                fig.add_trace(go.Scatter(
                                    x=results['best_match']['x_synth'],
                                    y=results['best_match']['y_synth'],
                                    mode='lines',
                                    name='Best Match',
                                    line=dict(color='red', width=2, dash='dot'))
                                )
                                
                                # Add detection threshold line
                                fig.add_hline(y=detection_threshold, line=dict(color="yellow", width=2, dash="dash"),
                                             annotation_text=f"{sigma_threshold}σ Detection Threshold", 
                                             annotation_position="bottom right")
                                
                                # Add baseline mean line
                                fig.add_hline(y=input_mean, line=dict(color="green", width=1, dash="dot"),
                                            annotation_text="Mean Baseline", 
                                            annotation_position="bottom right")
                                
                                # Mark significant coincident peaks
                                if coincident_peaks:
                                    input_peak_freqs = [p['input_freq'] for p in coincident_peaks]
                                    input_peak_ints = [p['input_intensity'] for p in coincident_peaks]
                                    sigma_levels = [p['sigma_level'] for p in coincident_peaks]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=input_peak_freqs,
                                        y=input_peak_ints,
                                        mode='markers+text',
                                        name=f'Coincident Peaks (>${sigma_threshold}\sigma$)',
                                        marker=dict(
                                            color=sigma_levels,
                                            colorscale='Rainbow',
                                            size=12,
                                            colorbar=dict(title='Sigma Level'),
                                        text=[f"{s:.1f}σ" for s in sigma_levels],
                                        textposition="top center"
                                    ))
                                
                                fig.update_layout(
                                    xaxis_title='Frequency (GHz)',
                                    yaxis_title='Intensity',
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    height=600
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)

                            # (Mantener el resto de las pestañas igual)

                    os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

else:
    st.info("Please upload an input spectrum file to begin analysis.")

# Instructions (mantener igual)