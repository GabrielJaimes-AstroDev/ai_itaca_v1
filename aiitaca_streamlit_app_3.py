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
import time

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
        background-color: #15181c !important;
    }
    
    /* Texto general en blanco/tonos claros */
    body, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stSelectbox, .stNumberInput, .stTextInput,
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    /* Sidebar blanco con texto oscuro */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Botones azules */
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
    
    /* Panel de descripción */
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
    
    /* Espacio para los botones de información */
    .info-buttons-container {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Configuración del gráfico Plotly */
    .plotly-graph-div {
        background-color: #0D0F14 !important;
    }
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

# Project description
st.markdown("""
<div class="description-panel" style="text-align: justify;">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring...
</div>
""", unsafe_allow_html=True)

# === CONFIGURATION ===
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1PyAGVOum6MWE_2PDqysvrr_dg_acg-v8?usp=drive_link"
TEMP_MODEL_DIR = "downloaded_models"

if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

@st.cache_data(show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    try:
        # (Código de descarga permanece igual)
        return model_files, data_files, True
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading models: {str(e)}")
        return [], [], False

# === SIDEBAR ===
st.sidebar.title("Configuration")

model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

input_file = st.sidebar.file_uploader(
    "Input Spectrum File ( . | .txt | .dat | .fits | .spec )",
    type=None,
    help="Drag and drop file here ( . | .txt | .dat | .fits | .spec ). Limit 200MB per file"
)

st.sidebar.subheader("Peak Matching Parameters")
sigma_emission = st.sidebar.slider("Sigma Emission", 0.1, 5.0, 1.5, step=0.1, key="sigma_emission")
window_size = st.sidebar.slider("Window Size", 1, 20, 3, step=1)
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 2.0, step=0.1, key="sigma_threshold")
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

# Botones de información con mejor alineación
st.markdown('<div class="info-buttons-container"></div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    params_tab = st.button("📝 Parameters Explanation", key="params_btn")
with col2:
    flow_tab = st.button("📊 Flow of Work Diagram", key="flow_btn")

def show_controls_and_plot():
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        
        # Obtener valores actuales de los sliders
        current_sigma_emission = st.session_state.sigma_emission
        current_sigma_threshold = st.session_state.sigma_threshold
        
        # Controles para mostrar/ocultar las líneas
        col1, col2 = st.columns(2)
        with col1:
            show_sigma = st.checkbox("Visualize Sigma Emission", value=True, key="show_sigma")
        with col2:
            show_threshold = st.checkbox("Visualize Sigma Threshold", value=True, key="show_threshold")
        
        # Crear gráfico base
        fig = go.Figure(st.session_state['base_fig'])
        
        # Calcular valores para las líneas
        input_spec = results['input_spec']
        noise_std = np.std(input_spec)
        
        # Añadir líneas según los checkboxes
        if show_sigma:
            sigma_line_y = current_sigma_emission * noise_std
            fig.add_hline(y=sigma_line_y, line_dash="dot",
                         annotation_text=f"Sigma Emission: {current_sigma_emission}",
                         annotation_position="bottom right",
                         line_color="yellow")
        
        if show_threshold:
            threshold_line_y = current_sigma_threshold * noise_std
            fig.add_hline(y=threshold_line_y, line_dash="dot",
                         annotation_text=f"Sigma Threshold: {current_sigma_threshold}",
                         annotation_position="bottom left",
                         line_color="cyan")
        
        # Mostrar gráfico actualizado
        st.plotly_chart(fig, use_container_width=True, key="main_plot")

if input_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name

    if not model_files:
        st.error("No trained models were found in Google Drive.")
    else:
        selected_model = st.selectbox("Select Molecule Model", model_files)
        analyze_btn = st.button("Analyze Spectrum")

        if analyze_btn:
            try:
                # Progreso del análisis
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                def update_analysis_progress(step, total_steps=6):
                    progress = int((step/total_steps)*100)
                    progress_bar.progress(progress)
                    steps = ["Loading model...", "Processing spectrum...", 
                            "Detecting peaks...", "Matching with database...",
                            "Calculating parameters...", "Generating visualizations..."]
                    progress_text.text(f"🔍 Analyzing spectrum... {steps[step-1]} ({progress}%)")
                
                # Análisis (simplificado para el ejemplo)
                results = analyze_spectrum(tmp_path, config, selected_model)
                
                # Guardar resultados
                st.session_state['analysis_results'] = results
                
                # Crear gráfico base
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['input_freq'],
                    y=results['input_spec'],
                    mode='lines',
                    name='Input Spectrum',
                    line=dict(color='white', width=2))
                
                fig.add_trace(go.Scatter(
                    x=results['best_match']['x_synth'],
                    y=results['best_match']['y_synth'],
                    mode='lines',
                    name='Best Match',
                    line=dict(color='red', width=2))
                
                fig.update_layout(
                    plot_bgcolor='#0D0F14',
                    paper_bgcolor='#0D0F14',
                    margin=dict(l=50, r=50, t=60, b=50),
                    xaxis_title='Frequency (GHz)',
                    yaxis_title='Intensity (K)',
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
                st.success("Analysis completed successfully!")
                
                # Mostrar controles y gráfico
                show_controls_and_plot()
                
                # Pestañas adicionales
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Molecule Best Match", 
                    "Peak Matching", 
                    "CNN Training", 
                    "Top Selection: LogN", 
                    "Top Selection: Tex"
                ])
                
                # (Contenido de las pestañas permanece igual)
                
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Mostrar controles si el análisis ya se realizó
if 'analysis_results' in st.session_state:
    show_controls_and_plot()

# Instructions
st.sidebar.markdown("""
**Instructions:**
1. Upload your input spectrum file
2. Adjust the peak matching parameters
3. Select the model to use
4. Click 'Analyze Spectrum'
""")
