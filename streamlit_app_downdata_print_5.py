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

# Configuración de la página con estilo personalizado y emoji de telescopio
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭"  # Emoji de telescopio
)

# === ESTILOS CSS PERSONALIZADOS ===
st.markdown("""
<style>
    /* Color negro para todo el texto */
    body, .stApp, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stButton>button, .stSelectbox, 
    .stFileUploader, .stNumberInput, .stTextInput {
        color: #000000 !important;
    }
    
    /* Sidebar blanco con texto negro */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Botones con bordes azules */
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
    
    /* Títulos y encabezados */
    h1 {
        color: #1E88E5 !important;
        border-bottom: 2px solid #1E88E5 !important;
        padding-bottom: 10px !important;
    }
    
    /* Panel descriptivo mejorado */
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
    
    /* Sliders y controles */
    .stSlider .thumb {
        background-color: #1E88E5 !important;
    }
    .stSlider .track {
        background-color: #BBDEFB !important;
    }
    
    /* Pestañas */
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
    
    /* Estilo para el file uploader - Cambiado a plomo claro */
    .stFileUploader>div>div>div>div {
        background-color: #E0E0E0 !important;
        border-color: #B0B0B0 !important;
        color: #000000 !important;
    }
    .stFileUploader>div>div>div>div:hover {
        background-color: #D0D0D0 !important;
    }
    .stFileUploader>div>section>div>div>div>span {
        color: #000000 !important;
    }
    
    /* Ocultar títulos de pestañas */
    .tab-content h2, .tab-content h3, .tab-content h4 {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# === CABECERA CON IMAGEN Y DESCRIPCIÓN ===
# Imagen en ancho completo arriba
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

# Columnas para el título y descripción debajo de la imagen
col1, col2 = st.columns([1, 3])
with col1:
    # Espacio vacío para mantener el layout (podrías poner un logo pequeño aquí si lo deseas)
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

# Descripción del proyecto
st.markdown("""
<div class="description-panel">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.

This project, <strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
</div>
""", unsafe_allow_html=True)

# === CONFIGURACIÓN ===
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ng3gqWqPW9v7ZtbXHx6GLOHXj8fDGh8Q?usp=sharing"
TEMP_MODEL_DIR = "downloaded_models"

# Crea carpeta temporal para los modelos si no existe
if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

# Función para descargar archivos de una carpeta pública de Google Drive usando gdown
@st.cache_data(show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    # Verificar si ya existen archivos .keras y .npz
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    if model_files and data_files:
        return model_files, data_files, True

    # Si no existen, descargar desde Google Drive
    try:
        with st.spinner("📥 Downloading models from Google Drive..."):
            gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
        
        # Volver a listar los archivos descargados
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

# === SIDEBAR ===
st.sidebar.title("Configuration")

# Descargar modelos desde Drive
model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

# File uploader en el sidebar con el nuevo estilo
input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=None,
    help="Drag and drop file here. Limit 200MB per file"
)

# Peak matching sliders
st.sidebar.subheader("Peak Matching Parameters")
sigma_emission = st.sidebar.slider("Sigma Emission", 0.1, 5.0, 1.5, step=0.1)
window_size = st.sidebar.slider("Window Size", 1, 20, 3, step=1)
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 2.0, step=0.1)
fwhm_ghz = st.sidebar.slider("FWHM (GHz)", 0.01, 0.5, 0.05, step=0.01)
tolerance_ghz = st.sidebar.slider("Tolerance (GHz)", 0.01, 0.5, 0.1, step=0.01)
min_peak_height_ratio = st.sidebar.slider("Min Peak Height Ratio", 0.1, 1.0, 0.3, step=0.05)
top_n_lines = st.sidebar.slider("Top N Lines", 5, 100, 30, step=5)
top_n_similar = st.sidebar.slider("Top N Similar", 50, 2000, 500, step=50)

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
        selected_model = st.selectbox("Select Model", model_files)

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

                        # Análisis del espectro
                        results = analyze_spectrum(
                            tmp_path, model, train_data, train_freq,
                            filenames, headers, train_logn, train_tex,
                            config, mol_name
                        )

                        st.success("Analysis completed successfully!")

                        # Crear pestañas para los gráficos
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
                                # Box con parámetros físicos del best match
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"""
                                    <div style="
                                        background-color: #f0f8ff;
                                        padding: 15px;
                                        border-radius: 10px;
                                        border-left: 5px solid #1E88E5;
                                        margin-bottom: 20px;
                                    ">
                                        <h4 style="color: #1E88E5; margin-top: 0;">Detection Physical Parameters</h4>
                                        <p><strong>LogN:</strong> {results['best_match']['logn']:.2f} cm⁻²</p>
                                        <p><strong>Tex:</strong> {results['best_match']['tex']:.2f} K</p>
                                        <p><strong>File:</strong> {results['best_match']['filename']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Crear figura interactiva con Plotly
                                fig = go.Figure()
                                
                                # Añadir espectro de entrada
                                fig.add_trace(go.Scatter(
                                    x=results['input_freq'],
                                    y=results['input_spec'],
                                    mode='lines',
                                    name='Input Spectrum',
                                    line=dict(color='white', width=2))
                                )
                                
                                # Añadir mejor coincidencia
                                fig.add_trace(go.Scatter(
                                    x=results['best_match']['x_synth'],
                                    y=results['best_match']['y_synth'],
                                    mode='lines',
                                    name='Best Match',
                                    line=dict(color='red', width=2))
                                )
                                
                                # Configuración del layout
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
                            if results['best_match']:
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

# Add some instructions
st.sidebar.markdown("""
**Instructions:**
1. Select the directory containing the trained models
2. Upload your input spectrum file (.txt or .dat)
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