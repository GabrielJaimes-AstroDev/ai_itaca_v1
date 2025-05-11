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
    /* Black text for all elements */
    body, .stApp, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stButton>button, .stSelectbox, 
    .stFileUploader, .stNumberInput, .stTextInput {
        color: #000000 !important;
    }
    
    /* White sidebar with black text */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Blue-bordered buttons */
    .stButton>button {
        border: 2px solid #1E88E5 !important;
        color: white !important;
        background-color: #1E88E5 !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s !important;
    }
    .stButton>button:hover {
        border: 2px solid #0D47A1 !important;
        background-color: #0D47A1 !important;
        color: white !important;
    }
    
    /* Titles and headers */
    h1 {
        color: #1E88E5 !important;
        border-bottom: 2px solid #1E88E5 !important;
        padding-bottom: 10px !important;
    }
    
    /* Enhanced description panel */
    .description-panel {
        text-align: justify;
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .description-panel p {
        margin-bottom: 15px;
        line-height: 1.6;
    }
    
    /* Sliders and controls */
    .stSlider .thumb {
        background-color: #1E88E5 !important;
    }
    .stSlider .track {
        background-color: #BBDEFB !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        color: #000000;
        background-color: #E3F2FD;
        border-radius: 5px 5px 0 0;
        border: 1px solid #BBDEFB;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    
    /* New styles for file uploader (light background) */
    .stFileUploader>div>div>div>div {
        background-color: #F5F5F5 !important;
        border: 1px solid #B0B0B0 !important;
        color: #000000 !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    /* Hover color change */
    .stFileUploader>div>div>div>div:hover {
        background-color: #E0E0E0 !important;
        border-color: #909090 !important;
    }
    /* Style for "Drag and drop file here..." text */
    .stFileUploader>div>section>div>div>div>span {
        color: #000000 !important;
        font-size: 14px !important;
    }
    /* Style for "Browse Files" button */
    .stFileUploader>div>section>div>button {
        background-color: #1E88E5 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        transition: background-color 0.3s !important;
    }
    /* Button hover effect */
    .stFileUploader>div>section>div>button:hover {
        background-color: #0D47A1 !important;
    }
            
    /* New added style - Button text color */
    .stFileUploader>div>section>div>button span {
        color: white !important;
    }
        
    /* Hide tab content titles */
    .tab-content h2, .tab-content h3, .tab-content h4 {
        display: none !important;
    }
    
    /* Error message style */
    .no-molecule-detected {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Warning message style */
    .weak-detection {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Success message style */
    .molecule-detected {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER WITH IMAGE AND DESCRIPTION ===
st.image("NGC6523_BVO_2.jpg", use_container_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown("""
    <style>
        .main-title {
            color: white !important;
            font-size: 2.5rem !important;
            font-weight: bold !important;
        }
        .subtitle {
            color: white !important;
            font-size: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

# Project description
st.markdown("""
<div class="description-panel">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.

<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
</div>
""", unsafe_allow_html=True)

# === CONFIGURATION ===
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1zlnkEoRvHR1CoK9hXxD0Jy4JIKF5Uybz?usp=sharing"
TEMP_MODEL_DIR = "downloaded_models"

if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

@st.cache_data(show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    if model_files and data_files:
        return model_files, data_files, True

    try:
        with st.spinner("📥 Downloading models from Google Drive..."):
            gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
        
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
        data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
        
        if model_files and data_files:
            st.sidebar.success("✅ Models downloaded successfully!")
        else:
            st.sidebar.error("❌ No models found in the specified folder")
            
        return model_files, data_files, True
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading models: {str(e)}")
        return [], [], False

def find_coincident_peaks(input_freq, input_spec, best_match_freq, best_match_spec, tolerance=0.1):
    """Find coincident peaks between input spectrum and best match spectrum"""
    # Find peaks in input spectrum
    input_peaks, _ = find_peaks(input_spec, height=np.mean(input_spec) + np.std(input_spec))
    input_peak_freqs = input_freq[input_peaks]
    input_peak_ints = input_spec[input_peaks]
    
    # Find peaks in best match spectrum
    best_match_peaks, _ = find_peaks(best_match_spec, height=np.mean(best_match_spec) + np.std(best_match_spec))
    best_match_peak_freqs = best_match_freq[best_match_peaks]
    best_match_peak_ints = best_match_spec[best_match_peaks]
    
    # Find coincident peaks within tolerance
    coincident_peaks = []
    for i, input_f in enumerate(input_peak_freqs):
        for j, match_f in enumerate(best_match_peak_freqs):
            if abs(input_f - match_f) <= tolerance:
                coincident_peaks.append({
                    'input_freq': input_f,
                    'input_intensity': input_peak_ints[i],
                    'match_freq': match_f,
                    'match_intensity': best_match_peak_ints[j],
                    'delta_freq': abs(input_f - match_f)
                })
                break
    
    return coincident_peaks

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

def evaluate_molecule_presence(coincident_peaks, sigma_threshold=1.0):
    """Evaluate if molecule is present based on coincident peaks"""
    if not coincident_peaks:
        return {
            'present': False,
            'confidence': 'none',
            'message': 'No coincident peaks found between input and best match',
            'num_peaks': 0
        }
    
    # Calculate mean and std of input spectrum baseline (non-peak regions)
    input_ints = np.array([p['input_intensity'] for p in coincident_peaks])
    mean_intensity = np.mean(input_ints)
    std_intensity = np.std(input_ints)
    
    # Count significant peaks (>sigma_threshold)
    significant_peaks = [p for p in coincident_peaks 
                        if p['input_intensity'] > mean_intensity + sigma_threshold * std_intensity]
    
    num_sig_peaks = len(significant_peaks)
    
    if num_sig_peaks == 0:
        return {
            'present': False,
            'confidence': 'none',
            'message': 'No coincident peaks above detection threshold (1σ)',
            'num_peaks': 0
        }
    elif num_sig_peaks < 3:
        return {
            'present': True,
            'confidence': 'weak',
            'message': f'Only {num_sig_peaks} coincident peak(s) above detection threshold (1σ) - weak detection',
            'num_peaks': num_sig_peaks
        }
    else:
        return {
            'present': True,
            'confidence': 'strong',
            'message': f'{num_sig_peaks} coincident peaks above detection threshold (1σ) - confident detection',
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
sigma_emission = st.sidebar.slider("Sigma Emission", 0.1, 5.0, 1.5, step=0.1)
window_size = st.sidebar.slider("Window Size", 1, 20, 3, step=1)
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 2.0, step=0.1)
fwhm_ghz = st.sidebar.slider("FWHM (GHz)", 0.01, 0.5, 0.05, step=0.01)
tolerance_ghz = st.sidebar.slider("Tolerance (GHz)", 0.01, 0.5, 0.1, step=0.01)
min_peak_height_ratio = st.sidebar.slider("Min Peak Height Ratio", 0.1, 1.0, 0.3, step=0.05)
top_n_lines = st.sidebar.slider("Top N Lines", 5, 100, 30, step=5)
top_n_similar = st.sidebar.slider("Top N Similar", 50, 2000, 800, step=50)

config = {
    'trained_models_dir': TEMP_MODEL_DIR,
    'peak_matching': {
        'sigma_emission': sigma_emission,
        'window_size': window_size,
        'sigma_threshold': sigma_threshold,
        'fwhm_ghz': fwhm_ghz,
        'tolerance_ghz': tolerance_ghz,
        'min_peak_height_ratio': min_peak_height_ratio,
        'top_n_lines': top_n_lines,
        'debug': True,
        'top_n_similar': top_n_similar
    }
}

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

                        # Find coincident peaks between input and best match
                        if results.get('best_match'):
                            coincident_peaks = find_coincident_peaks(
                                results['input_freq'], 
                                results['input_spec'],
                                results['best_match']['x_synth'],
                                results['best_match']['y_synth'],
                                tolerance=tolerance_ghz
                            )
                            
                            # Evaluate molecule presence based on coincident peaks
                            presence = evaluate_molecule_presence(coincident_peaks, sigma_threshold=1.0)
                            
                            # Calculate errors from coincident peaks
                            peak_errors = calculate_errors_from_coincident_peaks(coincident_peaks)
                            
                            # Display detection result
                            if presence['present']:
                                if presence['confidence'] == 'strong':
                                    st.markdown(f"""
                                    <div class="molecule-detected">
                                        <h4>✅ Molecule Detected (Confident)</h4>
                                        <p>{presence['message']}</p>
                                        <p><strong>Coincident Peaks:</strong> {presence['num_peaks']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="weak-detection">
                                        <h4>⚠️ Molecule Detected (Weak)</h4>
                                        <p>{presence['message']}</p>
                                        <p><strong>Coincident Peaks:</strong> {presence['num_peaks']}</p>
                                        <p>Results should be interpreted with caution</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="no-molecule-detected">
                                    <h4>❌ Molecule Not Detected</h4>
                                    <p>{presence['message']}</p>
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
                            if results.get('best_match'):
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Display parameters with errors if available
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
                                            <p><strong>File (Top CNN Train):</strong> {results['best_match']['filename']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style="
                                            background-color: #f0f8ff;
                                            padding: 15px;
                                            border-radius: 10px;
                                            border-left: 5px solid #1E88E5;
                                            margin-bottom: 20px;
                                        ">
                                            <h4 style="color: #1E88E5; margin-top: 0;">Detection of Physical Parameters</h4>
                                            <p><strong>LogN:</strong> {results['best_match']['logn']:.2f} cm⁻²</p>
                                            <p><strong>Tex:</strong> {results['best_match']['tex']:.2f} K</p>
                                            <p><strong>No coincident peaks found</strong></p>
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
                                    line=dict(color='red', width=2))
                                )
                                
                                # Mark coincident peaks if any
                                if coincident_peaks:
                                    input_peak_freqs = [p['input_freq'] for p in coincident_peaks]
                                    input_peak_ints = [p['input_intensity'] for p in coincident_peaks]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=input_peak_freqs,
                                        y=input_peak_ints,
                                        mode='markers',
                                        name='Coincident Peaks',
                                        marker=dict(
                                            color='yellow',
                                            size=10,
                                            line=dict(width=1, color='black')
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

                        with tab1:
                            if results['best_match']:
                                st.pyplot(plot_summary_comparison(
                                    results['input_freq'], results['input_spec'],
                                    results['best_match'], tmp_path
                                ))

                        with tab2:
                            if results['best_match'] and coincident_peaks:
                                # Create a zoomed plot focusing on coincident peaks
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                # Plot input spectrum
                                ax.plot(results['input_freq'], results['input_spec'], 
                                       'k-', linewidth=1.5, label='Input Spectrum')
                                
                                # Plot best match
                                ax.plot(results['best_match']['x_synth'], results['best_match']['y_synth'],
                                       'r-', linewidth=1.5, alpha=0.7, label='Best Match')
                                
                                # Mark coincident peaks
                                for peak in coincident_peaks:
                                    ax.axvline(x=peak['input_freq'], color='blue', linestyle='--', alpha=0.3)
                                    ax.plot(peak['input_freq'], peak['input_intensity'], 
                                            'yo', markersize=8, markeredgecolor='black')
                                
                                ax.set_xlabel('Frequency (GHz)')
                                ax.set_ylabel('Intensity')
                                ax.set_title('Coincident Peaks Comparison')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                            elif results['best_match']:
                                st.pyplot(plot_zoomed_peaks_comparison(
                                    results['input_spec'], results['input_freq'],
                                    results['best_match']
                                ))

                        with tab3:
                            st.pyplot(plot_best_matches(
                                results['train_logn'], results['train_tex'],
                                results['similarities'], results['distances'],
                                results['closest_idx_sim'], results['closest_idx_dist'],
                                results['train_filenames'], results['input_logn']
                            ))

                        with tab4:
                            st.pyplot(plot_tex_metrics(
                                results['train_tex'], results['train_logn'],
                                results['similarities'], results['distances'],
                                results['top_similar_indices'],
                                results['input_tex'], results['input_logn']
                            ))

                        with tab5:
                            st.pyplot(plot_similarity_metrics(
                                results['train_logn'], results['train_tex'],
                                results['similarities'], results['distances'],
                                results['top_similar_indices'],
                                results['input_logn'], results['input_tex']
                            ))

                    os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

else:
    st.info("Please upload an input spectrum file to begin analysis.")

# Instructions
st.sidebar.markdown("""
**Instructions:**
1. Select the directory containing the trained models
2. Upload your input spectrum file 
3. Adjust the peak matching parameters as needed
4. Select the model to use for analysis
5. Click 'Analyze Spectrum' to run the analysis

**Detection Criteria:**
- No coincident peaks >1σ: Molecule not detected
- 1-2 coincident peaks >1σ: Weak detection (caution)
- 3+ coincident peaks >1σ: Confident detection

**Interactive Plot Controls:**
- 🔍 Zoom: Click and drag to select area
- 🖱️ Hover: View exact values
- 🔄 Reset: Double-click
- 🏎️ Pan: Shift+click+drag
- 📊 Range Buttons: Quick zoom to percentage ranges
""")