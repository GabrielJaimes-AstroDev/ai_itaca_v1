import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from core_functions import *
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
import gdown
import shutil
import time
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# === CUSTOM CSS STYLES ===
st.markdown("""
<style>
    /* Fondo principal color plomo oscuro y texto claro */
    .stApp, .main .block-container, body {
        background-color: #15181c !important;  /* Color plomo oscuro */
    }
    
    /* Texto general en blanco/tonos claros */
    body, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stSelectbox, .stNumberInput, .stTextInput,
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    /* Sidebar blanco con texto oscuro (sin cambios) */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Botones azules SOLO dentro de Molecular Analyzer */
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
    }
    
    /* Títulos y encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #1E88E5 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    h1 {
        border-bottom: 2px solid #1E88E5 !important;
        padding-bottom: 10px !important;
    }
    
    /* Panel de descripción adaptado */
    .description-panel {
        text-align: justify;
        background-color: white !important;
        color: black !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .description-panel p {
        margin-bottom: 15px;
        line-height: 1.6;
        color: #000000 !important;
    }
    .description-panel strong {
        color: #000000 !important;
        font-weight: bold;
    }
    
    /* Sliders y controles */
    .stSlider .thumb {
        background-color: #1E88E5 !important;
    }
    .stSlider .track {
        background-color: #5F9EA0 !important;  /* Tonos que combinan con plomo */
    }
    
    /* Pestañas personalizadas */
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        color: #FFFFFF !important;
        background-color: #81acde !important;  /* Tono intermedio */
        border-radius: 5px 5px 0 0;
        border: 1px solid #1E88E5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }    
    
    /* File uploader adaptado */
    .stFileUploader>div>div>div>div {
        background-color: #E5E7E9 !important;  /* Plomo muy claro */
        border: 2px solid #B0E0E6 !important;
        color: #000000 !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    .stFileUploader>div>div>div>div:hover {
        background-color: #D5D8DC !important;  /* Un poco más oscuro al hover */
        border-color: #1E88E5 !important;
    }
    .stFileUploader>div>section>div>div>div>span {
        color: #000000 !important;
        font-size: 14px !important;
    }
    .stFileUploader>div>section>div>button {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .stFileUploader>div>section>div>button:hover {
        background-color: #0D47A1 !important;
    }
    
    /* Ajustes para gráficos y visualizaciones */
    .stPlotlyChart, .stDataFrame {
        background-color: #1e88e5 !important;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Mejoras para inputs y selects */
    .stTextInput input, .stNumberInput input, 
    .stSelectbox select {
        background-color: #1e88e5 !important;
        color: white !important;
        border: 1px solid #5F9EA0 !important;
    }
    
    /* Ocultar títulos de contenido de pestañas */
    .tab-content h2, .tab-content h3, .tab-content h4 {
        display: none !important;
    }
    
    /* Estilo para los valores de Physical Parameters */
    .physical-params {
        color: #000000 !important;
        font-size: 1.1rem !important;
        margin: 5px 0 !important;
    }
    
    /* Panel azul claro para el resumen */
    .summary-panel {
        background-color: #FFFFFF !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    
    /* Fondo del gráfico interactivo */
    .js-plotly-plot .plotly, .plot-container {
        background-color: #0D0F14 !important;
    }
    
    /* Configuración del gráfico Plotly */
    .plotly-graph-div {
        background-color: #0D0F14 !important;
    }
    
    /* Nuevos estilos para paneles de información */
    .info-panel {
        background-color: white !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;
    }
    .info-panel h3 {
        color: #1E88E5 !important;
        margin-top: 0;
    }
    .info-panel img {
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        margin: 10px 0;
    }
    .pro-tip {
        background-color: #f0f7ff !important;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
        margin-top: 15px;
    }
    .pro-tip p {
        margin: 0;
        font-size: 0.9em;
        color: #333 !important;
    }
    
    /* Estilo para la barra de progreso */
    .progress-bar {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* Espacio para los botones de información */
    .info-buttons-container {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Contenedor de botones ajustado */
    .buttons-container {
        display: flex;
        gap: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Estilos para el visualizador de cubos */
    .cube-controls {
        background-color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
    }
    .cube-status {
        background-color: #f8f9fa !important;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-family: monospace;
        color: #000000 !important;
    }
    .spectrum-display {
        background-color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #1E88E5;
        color: #000000 !important;
    }
    
    /* Estilo para la tabla de training dataset */
    .training-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .training-table thead tr {
        background-color: #1E88E5;
        color: #ffffff;
        text-align: left;
    }
    .training-table th,
    .training-table td {
        padding: 12px 15px;
        border: 1px solid #dddddd;
    }
    .training-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .training-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .training-table tbody tr:last-of-type {
        border-bottom: 2px solid #1E88E5;
    }
    .training-table tbody tr:hover {
        background-color: #f1f1f1;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# === FUNCIÓN PARA BLOQUEAR CONTROLES DURANTE PROCESAMIENTO ===
def disable_widgets():
    """Deshabilita todos los widgets cuando hay procesamiento en curso"""
    processing = st.session_state.get('processing', False)
    return processing

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
<div class="description-panel" style="text-align: justify;">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
</div>
""", unsafe_allow_html=True)

# === CONFIGURATION ===
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1zlnkEoRvHR1CoK9hXxD0Jy4JIKF5Uybz?usp=drive_link"
TEMP_MODEL_DIR = "downloaded_models"

if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

@st.cache_data(ttl=3600, show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    if model_files and data_files:
        return model_files, data_files, True

    try:
        st.session_state['processing'] = True
        progress_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)
        progress_text.text("📥 Preparing to download models...")
        
        file_count = 0
        try:
            file_count = 10  # Valor estimado para la simulación
        except:
            file_count = 10  # Valor por defecto si no podemos obtener el conteo real
            
        with st.spinner("📥 Downloading models from Google Drive..."):
            gdown.download_folder(
                folder_url, 
                output=output_dir, 
                quiet=True,  # Silenciamos la salida por consola
                use_cookies=False
            )
            for i in range(file_count):
                time.sleep(0.5)  # Pequeña pausa para simular descarga
                progress = int((i + 1) / file_count * 100)
                progress_bar.progress(progress)
                progress_text.text(f"📥 Downloading models... {progress}%")
        
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
        data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
        
        progress_bar.progress(100)
        progress_text.text("Process Completed")
        
        if model_files and data_files:
            st.sidebar.success("✅ Models downloaded successfully!")
        else:
            st.sidebar.error("❌ No models found in the specified folder")
            
        return model_files, data_files, True
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading models: {str(e)}")
        return [], [], False
    finally:
        st.session_state['processing'] = False

# === SIDEBAR ===
st.sidebar.title("Configuration")

model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

# Clear previous analysis when new file is uploaded
if 'prev_uploaded_file' not in st.session_state:
    st.session_state.prev_uploaded_file = None

current_uploaded_file = st.sidebar.file_uploader(
    "Input Spectrum File ( . | .txt | .dat | .fits | .spec )",
    type=None,
    help="Drag and drop file here ( . | .txt | .dat | .fits | .spec ). Limit 200MB per file",
    disabled=disable_widgets()
)

# === NEW UNIT SELECTION WIDGETS ===
st.sidebar.markdown("---")
st.sidebar.subheader("Units Configuration")

# Frequency units selection
freq_unit = st.sidebar.selectbox(
    "Frequency Units",
    ["GHz", "MHz", "kHz", "Hz"],
    index=0,
    help="Select the frequency units for the input spectrum",
    disabled=disable_widgets()
)

# Intensity units selection
intensity_unit = st.sidebar.selectbox(
    "Intensity Units",
    ["K", "Jy"],
    index=0,
    help="Select the intensity units for the input spectrum",
    disabled=disable_widgets()
)

# Conversion factors
freq_conversion = {
    "GHz": 1e9,
    "MHz": 1e6,
    "kHz": 1e3,
    "Hz": 1.0
}

intensity_conversion = {
    "K": 1.0,
    "Jy": 1.0  # Placeholder - actual conversion would depend on the specific context
}

if current_uploaded_file != st.session_state.prev_uploaded_file:
    # Clear previous analysis results
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    if 'analysis_done' in st.session_state:
        del st.session_state['analysis_done']
    if 'base_fig' in st.session_state:
        del st.session_state['base_fig']
    if 'input_spec' in st.session_state:
        del st.session_state['input_spec']
    
    st.session_state.prev_uploaded_file = current_uploaded_file
    st.rerun()

st.sidebar.subheader("Peak Matching Parameters")
sigma_emission = st.sidebar.slider("Sigma Emission", 0.1, 5.0, 1.5, step=0.1, key="sigma_emission_slider", disabled=disable_widgets())
window_size = st.sidebar.slider("Window Size", 1, 20, 3, step=1, disabled=disable_widgets())
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 2.0, step=0.1, key="sigma_threshold_slider", disabled=disable_widgets())
fwhm_ghz = st.sidebar.slider("FWHM (GHz)", 0.01, 0.5, 0.05, step=0.01, disabled=disable_widgets())
tolerance_ghz = st.sidebar.slider("Tolerance (GHz)", 0.01, 0.5, 0.1, step=0.01, disabled=disable_widgets())
min_peak_height_ratio = st.sidebar.slider("Min Peak Height Ratio", 0.1, 1.0, 0.3, step=0.05, disabled=disable_widgets())
top_n_lines = st.sidebar.slider("Top N Lines", 5, 100, 30, step=5, disabled=disable_widgets())
top_n_similar = st.sidebar.slider("Top N Similar", 50, 2000, 800, step=50, disabled=disable_widgets())

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
    },
    'units': {
        'frequency': freq_unit,
        'intensity': intensity_unit
    }
}

# === CUBE VISUALIZER FUNCTIONS ===
@st.cache_data(ttl=3600, max_entries=3, show_spinner="Loading ALMA cube...")
def load_alma_cube(file_path, max_mb=2048):
    """Load ALMA cube from FITS file with memory management"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb > max_mb:
        raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed ({max_mb} MB)")
    
    with fits.open(file_path) as hdul:
        cube_data = hdul[0].data
        header = hdul[0].header
        
        # Basic cube information
        n_chan = cube_data.shape[0] if len(cube_data.shape) == 3 else 1
        ra_size = cube_data.shape[-2] if len(cube_data.shape) >= 2 else 1
        dec_size = cube_data.shape[-1] if len(cube_data.shape) >= 2 else 1
        
        # Get frequency information from header
        try:
            freq0 = header['CRVAL3']
            dfreq = header['CDELT3']
            freq_axis = freq0 + dfreq * np.arange(n_chan)
        except:
            freq_axis = None
        
        cube_info = {
            'data': cube_data,
            'header': header,
            'n_chan': n_chan,
            'ra_size': ra_size,
            'dec_size': dec_size,
            'freq_axis': freq_axis,
            'file_size_mb': file_size_mb
        }
    
    return cube_info

def display_cube_info(cube_info):
    """Display basic information about the loaded cube"""
    st.markdown(f"""
    <div class="cube-status">
        <strong>Cube Information:</strong><br>
        Dimensions: {cube_info['data'].shape}<br>
        Channels: {cube_info['n_chan']}<br>
        RA size: {cube_info['ra_size']} pixels<br>
        Dec size: {cube_info['dec_size']} pixels<br>
        File size: {cube_info['file_size_mb']:.2f} MB<br>
    </div>
    """, unsafe_allow_html=True)

def extract_spectrum_from_region(cube_data, x_range, y_range):
    """Extract average spectrum from selected region"""
    if len(cube_data.shape) == 3:
        # For 3D cubes: average over spatial dimensions
        spectrum = np.mean(cube_data[:, y_range[0]:y_range[1], x_range[0]:x_range[1]], axis=(1, 2))
    else:
        # For 2D images: return None
        spectrum = None
    return spectrum

def create_spectrum_download(freq_axis, spectrum, freq_unit='GHz', intensity_unit='K'):
    """Create downloadable text file with spectrum data"""
    if freq_axis is None or spectrum is None:
        return None
    
    # Convert frequency based on selected unit
    if freq_unit == 'GHz':
        freq_values = freq_axis / 1e9
    elif freq_unit == 'MHz':
        freq_values = freq_axis / 1e6
    elif freq_unit == 'kHz':
        freq_values = freq_axis / 1e3
    else:  # Hz
        freq_values = freq_axis
    
    # Create the file content
    content = f"!xValues({freq_unit})\tyValues({intensity_unit})\n"
    for freq, val in zip(freq_values, spectrum):
        content += f"{freq:.8f}\t{val:.6f}\n"
    
    return content

# === MAIN TABS ===
tab_molecular, tab_cube = st.tabs(["Molecular Analyzer", "Cube Visualizer"])

with tab_molecular:
    # === MOLECULAR ANALYZER CONTENT ===
    st.title("Molecular Spectrum Analyzer | AI - ITACA")

    # PARAMETERS EXPLANATION
    st.markdown('<div class="buttons-container"></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([0.5, 0.5, 0.5, 0.5])
    with col1:
        params_tab = st.button("📝 Parameters Explanation", key="params_btn", 
                            help="Click to show parameters explanation",
                            disabled=disable_widgets())
    with col2:
        training_tab = st.button("📊 Training dataset", key="training_btn", 
                            help="Click to show training dataset information",
                            disabled=disable_widgets())
    with col3:
        flow_tab = st.button("📊 Flow of Work Diagram", key="flow_btn", 
                        help="Click to show the workflow diagram",
                        disabled=disable_widgets())
    with col4:
        Acknowledgments_tab = st.button("✅ Acknowledgments", key="Acknowledgments_tab", 
                        help="Click to show Acknowledgments",
                        disabled=disable_widgets())

    if params_tab:
        with st.container():
            st.markdown("""
            <div class="description-panel">
                <h3 style="text-align: center; margin-top: 0; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Technical Parameters Guide</h3>
                
            <div style="margin-bottom: 25px;">
            <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">🔬 Peak Detection</h4>
            <p><strong>Sigma Emission (1.5):</strong> Threshold for peak detection in standard deviations (σ) of the noise. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Higher values reduce false positives but may miss weak peaks. Typical range: 1.0-3.0</span></p>
            
            <p><strong>Window Size (3):</strong> Points in Savitzky-Golay smoothing kernel. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Odd integers only. Larger values smooth noise but blur close peaks.</span></p>
            
            <p><strong>Sigma Threshold (2.0):</strong> Minimum peak prominence (σ). 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Filters low-significance features. Critical for crowded spectra.</span></p>
            
            <p><strong>FWHM (0.05 GHz):</strong> Expected line width at half maximum. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Should match your instrument's resolution. Affects line fitting.</span></p>
            </div>
                
            <div style="margin-bottom: 25px;">
            <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">🔄 Matching Parameters</h4>
            <p><strong>Tolerance (0.1 GHz):</strong> Frequency matching window. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Accounts for Doppler shifts (±20 km/s at 100 GHz). Increase for broad lines.</span></p>
            
            <p><strong>Min Peak Ratio (0.3):</strong> Relative intensity cutoff. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Peaks below this fraction of strongest line are excluded. Range: 0.1-0.5.</span></p>
            </div>
            
            <div>
            <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">📊 Output Settings</h4>
            <p><strong>Top N Lines (30):</strong> Lines displayed in results. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Doesn't affect analysis quality, only visualization density.</span></p>
            
            <p><strong>Top N Similar (800):</strong> Synthetic spectra retained. 
            <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Higher values improve accuracy but increase runtime. Max: 2000.</span></p>
            </div>
            
            <div style="margin-top: 20px; padding: 12px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #1E88E5;">
            <p style="margin: 0; font-size: 0.9em;"><strong>Pro Tip:</strong> For ALMA data (high resolution), start with FWHM=0.05 GHz and Tolerance=0.05 GHz. For single-dish telescopes, try FWHM=0.2 GHz.</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    # TRAINING DATASET
    if training_tab:
        with st.container():
            st.markdown("""
                <div class="info-panel">
                    <h3 style="text-align: center; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Training Dataset Parameters</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <table class="training-table">
                <thead>
                    <tr>
                        <th>Molécule</th>
                        <th>Tex Range (K)</th>
                        <th>Tex Step</th>
                        <th>LogN Range (cm⁻²)</th>
                        <th>LogN Step</th>
                        <th>Frequency Range (GHz)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>CO</td>
                        <td>20 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                    <tr>
                        <td>SiO</td>
                        <td>20 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                    <tr>
                        <td>HCO⁺ v=0,1,2</td>
                        <td>20 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                    <tr>
                        <td>CH3CN</td>
                        <td>20 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                    <tr>
                        <td>HNC</td>
                        <td>20 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                    <tr>
                        <td>SO</td>
                        <td>20 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                    <tr>
                        <td>CH3OCHO_Yebes</td>
                        <td>20 - 350</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>20 - 50</td>
                    </tr>
                    <tr>
                        <td>CH3OCHO</td>
                        <td>120 - 380</td>
                        <td>5</td>
                        <td>12 - 19.2</td>
                        <td>0.1</td>
                        <td>80 - 300</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="pro-tip">
                <p><strong>Note:</strong> The training dataset was generated using LTE radiative transfer models under typical ISM conditions.</p>
            </div>
            """, unsafe_allow_html=True)

    # FLOW OF WORK
    if flow_tab:
        with st.container():
            st.markdown("""
                <div class="info-panel">
                    <h3 style="text-align: center; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Flow of Work Diagram</h3>
                </div>
            """, unsafe_allow_html=True)

            st.image("Flow_of_Work.jpg", use_container_width=True)

            st.markdown("""
                <div style="margin-top: 20px;">
                <h4 style="color: #1E88E5; margin-bottom: 10px;">Analysis Pipeline Steps:</h4>
                <ol style="color: white; padding-left: 20px;">
                <li><strong>Spectrum Input:</strong> Upload your observational spectrum</li>
                <li><strong>Pre-processing:</strong> Noise reduction and baseline correction</li>
                <li><strong>Peak Detection:</strong> Identify significant spectral features</li>
                <li><strong>Model Matching:</strong> Compare with synthetic spectra database</li>
                <li><strong>Parameter Estimation:</strong> Determine physical conditions (T<sub>ex</sub>, logN)</li>
                <li><strong>Visualization:</strong> Interactive comparison of observed vs synthetic spectra</li>
                </ol>
                </div>
                <div class="pro-tip">
                <p><strong>Note:</strong> The complete analysis typically takes 30-90 seconds depending on spectrum complexity and selected parameters.</p>
                </div>
            """, unsafe_allow_html=True)

    # ACKNOWLEDGMENTS
    if Acknowledgments_tab:
        with st.container():
            st.markdown("""
                <div class="info-panel">
                    <h3 style="text-align: center; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Project Acknowledgments</h3>
                </div>
            """, unsafe_allow_html=True)

            st.image("Acknowledgments.png", use_container_width=True)

            st.markdown("""<div class="description-panel" style="text-align: justify;">
            "The funding for these actions/grants and contracts comes from the European Union's Recovery and Resilience Facility-Next Generation, in the framework of the General Invitation of the Spanish Government's public business entity Red.es to participate in talent attraction and retention programmes within Investment 4 of Component 19 of the Recovery, Transformation and Resilience Plan".
            </div>
            """, unsafe_allow_html=True)

    if current_uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(current_uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if not model_files:
            st.error("No trained models were found in Google Drive.")
        else:
            selected_model = st.selectbox(
                "Select Molecule Model", 
                model_files,
                disabled=disable_widgets()
            )
            
            analyze_btn = st.button(
                "Analyze Spectrum",
                disabled=disable_widgets()
            )

            if analyze_btn:
                try:
                    st.session_state['processing'] = True
                    # Configurar la barra de progreso para el análisis
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def update_analysis_progress(step, total_steps=6):
                        progress = int((step / total_steps) * 100)
                        progress_bar.progress(progress)
                        steps = [
                            "Loading model...",
                            "Processing spectrum...",
                            "Detecting peaks...",
                            "Matching with database...",
                            "Calculating parameters...",
                            "Generating visualizations..."
                        ]
                        progress_text.text(f"🔍 Analyzing spectrum... {steps[step-1]} ({progress}%)")
                    
                    update_analysis_progress(1)
                    mol_name = selected_model.replace('_model.keras', '')

                    model_path = os.path.join(TEMP_MODEL_DIR, selected_model)
                    model = tf.keras.models.load_model(model_path)

                    update_analysis_progress(2)
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

                        update_analysis_progress(3)
                        results = analyze_spectrum(
                            tmp_path, model, train_data, train_freq,
                            filenames, headers, train_logn, train_tex,
                            config, mol_name
                        )

                        # Apply unit conversions to the results
                        if freq_unit != 'GHz':
                            results['input_freq'] = results['input_freq'] * 1e9 / freq_conversion[freq_unit]
                            results['best_match']['x_synth'] = results['best_match']['x_synth'] * 1e9 / freq_conversion[freq_unit]
                        
                        if intensity_unit != 'K':
                            results['input_spec'] = results['input_spec'] * intensity_conversion[intensity_unit]
                            results['best_match']['y_synth'] = results['best_match']['y_synth'] * intensity_conversion[intensity_unit]

                        update_analysis_progress(6)
                        st.success("Analysis completed successfully!")

                        st.session_state['analysis_results'] = results
                        st.session_state['analysis_done'] = True
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['input_freq'],
                            y=results['input_spec'],
                            mode='lines',
                            name='Input Spectrum',
                            line=dict(color='white', width=2)))
                        
                        fig.add_trace(go.Scatter(
                            x=results['best_match']['x_synth'],
                            y=results['best_match']['y_synth'],
                            mode='lines',
                            name='Best Match',
                            line=dict(color='red', width=2)))
                        
                        fig.update_layout(
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            margin=dict(l=50, r=50, t=60, b=50),
                            xaxis_title=f'Frequency ({freq_unit})',
                            yaxis_title=f'Intensity ({intensity_unit})',
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=600,
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#3A3A3A'),
                            yaxis=dict(gridcolor='#3A3A3A')
                        )
                        
                        st.session_state['base_fig'] = fig
                        st.session_state['input_spec'] = results['input_spec']

                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                finally:
                    st.session_state['processing'] = False

            # Mostrar pestañas si el análisis está completo
            if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
                tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Interactive Summary", 
                    "Molecule Best Match", 
                    "Peak Matching", 
                    "CNN Training", 
                    "Top Selection: LogN", 
                    "Top Selection: Tex"
                ])

                with tab0:
                    results = st.session_state['analysis_results']
                    st.markdown(f"""
                    <div class="summary-panel">
                        <h4 style="color: #1E88E5; margin-top: 0;">Detection of Physical Parameters</h4>
                        <p class="physical-params"><strong>LogN:</strong> {results['best_match']['logn']:.2f} cm⁻²</p>
                        <p class="physical-params"><strong>Tex:</strong> {results['best_match']['tex']:.2f} K</p>
                        <p class="physical-params"><strong>File (Top CNN Train):</strong> {results['best_match']['filename']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        show_sigma = st.checkbox("Visualize Sigma Emission", value=True, 
                                            key="show_sigma_checkbox",
                                            disabled=disable_widgets())
                    with col2:
                        show_threshold = st.checkbox("Visualize Sigma Threshold", value=True,
                                                key="show_threshold_checkbox",
                                                disabled=disable_widgets())
                    
                    fig = go.Figure(st.session_state['base_fig'])
                    
                    if show_sigma:
                        sigma_line_y = sigma_emission * np.std(st.session_state['input_spec'])
                        fig.add_hline(y=sigma_line_y, line_dash="dot",
                                    annotation_text=f"Sigma Emission: {sigma_emission}",
                                    annotation_position="bottom right",
                                    line_color="yellow")
                    
                    if show_threshold:
                        threshold_line_y = sigma_threshold * np.std(st.session_state['input_spec'])
                        fig.add_hline(y=threshold_line_y, line_dash="dot",
                                    annotation_text=f"Sigma Threshold: {sigma_threshold}",
                                    annotation_position="bottom left",
                                    line_color="cyan")
                    
                    # Mostrar el gráfico
                    st.plotly_chart(fig, use_container_width=True, key="main_plot")

                with tab1:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_summary_comparison(
                            results['input_freq'], results['input_spec'],
                            results['best_match'], tmp_path
                        ))

                with tab2:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_zoomed_peaks_comparison(
                            results['input_spec'], results['input_freq'],
                            results['best_match']
                        ))

                with tab3:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_best_matches(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['closest_idx_sim'], results['closest_idx_dist'],
                            results['train_filenames'], results['input_logn']
                        ))

                with tab4:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_tex_metrics(
                            results['train_tex'], results['train_logn'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_tex'], results['input_logn']
                        ))

                with tab5:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_similarity_metrics(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_logn'], results['input_tex']
                        ))

    # Instructions
    st.sidebar.markdown("""
    **Instructions:**
    1. Select the directory containing the trained models
    2. Upload your input spectrum file ( . | .txt | .dat | .fits | .spec )
    3. Adjust the peak matching parameters as needed
    4. Select the model to use for analysis
    5. Click 'Analyze Spectrum' to run the analysis

    **Interactive Plot Controls:**
    - 🔍 Zoom: Click and drag to select area
    - 🖱️ Hover: View exact values
    - 🔄 Reset: Double-click
    - 🏎️ Pan: Shift+click+drag
    - 📊 Range Buttons: Quick zoom to percentage ranges
    """)

with tab_cube:
    # === CUBE VISUALIZER CONTENT ===
    st.title("Cube Visualizer | AI-ITACA")
    st.markdown("""
    <div class="description-panel">
        <h3 style="text-align: center; margin-top: 0; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">3D Spectral Cube Visualization</h3>
        <p>Upload and visualize ALMA spectral cubes (FITS format) up to 2GB in size. Explore different channels and create integrated intensity maps.</p>
    </div>
    """, unsafe_allow_html=True)
    
    cube_file = st.file_uploader(
        "Upload ALMA Cube (FITS format)",
        type=["fits", "FITS"],
        help="Drag and drop ALMA cube FITS file here (up to 2GB)",
        disabled=disable_widgets()
    )
    
    if cube_file is not None:
        with st.spinner("Processing ALMA cube..."):
            st.session_state['processing'] = True
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp_cube:
                tmp_cube.write(cube_file.getvalue())
                tmp_cube_path = tmp_cube.name
            
            try:
                cube_info = load_alma_cube(tmp_cube_path)
                display_cube_info(cube_info)
                
                st.success("ALMA cube loaded successfully!")
                
                # Basic cube visualization controls
                st.markdown("""
                <div class="cube-controls">
                    <h4 style="color: #1E88E5; margin-top: 0;">Cube Visualization Controls</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    channel = st.slider(
                        "Select Channel",
                        0, cube_info['n_chan']-1, cube_info['n_chan']//2,
                        help="Navigate through spectral channels",
                        disabled=disable_widgets()
                    )
                    
                    st.markdown(f"""
                    <div class="cube-status">
                        <strong>Current Channel:</strong> {channel}<br>
                        {f"Frequency: {cube_info['freq_axis'][channel]/1e9:.4f} GHz" if cube_info['freq_axis'] is not None else ""}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="margin-top: 20px;">
                        <strong>Visualization Options:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    show_rms = st.checkbox("Show RMS noise level", value=True, disabled=disable_widgets())
                    scale = st.selectbox("Image Scale", ["Linear", "Log", "Sqrt"], index=0, disabled=disable_widgets())
                
                # Create interactive plot with Plotly
                if len(cube_info['data'].shape) == 3:
                    img_data = cube_info['data'][channel, :, :]
                else:
                    img_data = cube_info['data']
                
                if scale == "Log":
                    img_data = np.log10(img_data - np.nanmin(img_data) + 1)
                elif scale == "Sqrt":
                    img_data = np.sqrt(img_data - np.nanmin(img_data))
                
                # Create interactive figure
                fig = px.imshow(
                    img_data,
                    origin='lower',
                    color_continuous_scale='viridis',
                    labels={'color': 'Intensity (K)'},
                    title=f"Channel {channel}" + (f" ({cube_info['freq_axis'][channel]/1e9:.4f} GHz)" if cube_info['freq_axis'] is not None else "")
                )
                
                fig.update_layout(
                    plot_bgcolor='#0D0F14',
                    paper_bgcolor='#0D0F14',
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title="RA (pixels)",
                    yaxis_title="Dec (pixels)",
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#3A3A3A'),
                    yaxis=dict(gridcolor='#3A3A3A')
                )
                
                # Display the interactive plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Region selection and spectrum extraction
                st.markdown("""
                <div class="cube-controls">
                    <h4 style="color: #1E88E5; margin-top: 0;">Region Selection</h4>
                    <p>Click on the image to select a pixel or draw a rectangle to select a region.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # If we have a cube with spectral dimension
                if len(cube_info['data'].shape) == 3 and cube_info['freq_axis'] is not None:
                    # Extract spectrum from selected region
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_range = st.slider("X Range (pixels)", 0, cube_info['ra_size']-1, (0, cube_info['ra_size']-1), disabled=disable_widgets())
                    with col2:
                        y_range = st.slider("Y Range (pixels)", 0, cube_info['dec_size']-1, (0, cube_info['dec_size']-1), disabled=disable_widgets())
                    
                    spectrum = extract_spectrum_from_region(
                        cube_info['data'],
                        x_range,
                        y_range
                    )
                    
                    if spectrum is not None:
                        # Plot the spectrum with Plotly
                        fig_spec = go.Figure()
                        fig_spec.add_trace(go.Scatter(
                            x=cube_info['freq_axis']/1e9,
                            y=spectrum,
                            mode='lines',
                            line=dict(color='#1E88E5', width=2)
                        ))
                        
                        fig_spec.update_layout(
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            margin=dict(l=50, r=50, t=60, b=50),
                            xaxis_title='Frequency (GHz)',
                            yaxis_title='Intensity (K)',
                            hovermode='x unified',
                            height=400,
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#3A3A3A'),
                            yaxis=dict(gridcolor='#3A3A3A')
                        )
                        
                        st.markdown("""
                        <div class="spectrum-display">
                            <h4 style="color: #1E88E5; margin-top: 0;">Extracted Spectrum</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.plotly_chart(fig_spec, use_container_width=True)
                        
                        # Create download button for the spectrum
                        spectrum_content = create_spectrum_download(
                            cube_info['freq_axis'], 
                            spectrum,
                            freq_unit='GHz',
                            intensity_unit='K'
                        )
                        if spectrum_content:
                            st.download_button(
                                label="Download Spectrum as TXT",
                                data=spectrum_content,
                                file_name="extracted_spectrum.txt",
                                mime="text/plain",
                                disabled=disable_widgets()
                            )
                
                os.unlink(tmp_cube_path)
                
            except Exception as e:
                st.error(f"Error processing ALMA cube: {str(e)}")
                if os.path.exists(tmp_cube_path):
                    os.unlink(tmp_cube_path)
            finally:
                st.session_state['processing'] = False

# === CACHED FUNCTIONS ===
@st.cache_data(ttl=3600)
def load_model(_model_path):
    return tf.keras.models.load_model(_model_path)

@st.cache_data(ttl=3600)
def load_training_data(data_file):
    with np.load(data_file) as data:
        return {
            'train_freq': data['train_freq'],
            'train_data': data['train_data'],
            'train_logn': data['train_logn'],
            'train_tex': data['train_tex'],
            'headers': data['headers'],
            'filenames': data['filenames']
        }

@st.cache_data(ttl=3600)
def cached_analyze_spectrum(tmp_path, model, train_data, train_freq, filenames, headers, train_logn, train_tex, config, mol_name):
    return analyze_spectrum(
        tmp_path, model, train_data, train_freq,
        filenames, headers, train_logn, train_tex,
        config, mol_name
    )
