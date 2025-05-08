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

# Configuración de la página con estilo personalizado
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔬"
)

# === ESTILOS CSS PERSONALIZADOS ===
st.markdown("""
<style>
    /* Botones con bordes azules */
    .stButton>button {
        border: 2px solid #4CAF50;
        color: white;
        background-color: #4CAF50;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border: 2px solid #2E7D32;
        background-color: #2E7D32;
        color: white;
    }
    
    /* Títulos y encabezados */
    h1 {
        color: #1E88E5;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    /* Mejorar los sliders */
    .stSlider [data-testid="stMarkdownContainer"] {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# === CABECERA CON IMAGEN Y DESCRIPCIÓN ===
col1, col2 = st.columns([1, 3])
with col1:
    # Aquí puedes reemplazar con tu imagen (puede ser una URL o un archivo local)
    st.image("https://via.placeholder.com/300x150?text=AI-ITACA+Logo", width=300)
    
with col2:
    st.title("AI-ITACA | Molecular Spectrum Analyzer")
    st.markdown("**Artificial Intelligence Integral Tool for AstroChemical Analysis**")

# Descripción del proyecto
st.markdown("""
<div style="text-align: justify; background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;">
    <p>A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life.</p>
    
    <p>The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.</p>
    
    <p>This project, <strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new <strong>AI-based cutting-edge analysis software</strong> that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.</p>
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
        return model_files, data_files

    # Si no existen, descargar desde Google Drive
    gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)

    # Volver a listar los archivos descargados
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    return model_files, data_files


# === SIDEBAR ===
st.sidebar.title("Configuration")

# Descargar modelos desde Drive
st.sidebar.write("📥 Downloading Models for detection...")
model_files, data_files = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

# File input
input_file = st.sidebar.file_uploader("Input Spectrum File", type=None)

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
st.title("AI - ITACA | Molecular Spectrum Analyzer")

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

                        st.success("¡Análisis completado con éxito!")

                        # Mostrar gráficos como antes...
                        # (Aquí puedes copiar tus secciones de plot_similarity_metrics, plot_tex_metrics, etc.)

                        st.subheader("Similarity Metrics")
                        st.pyplot(plot_similarity_metrics(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_logn'], results['input_tex']
                        ))

                        st.subheader("Tex Metrics")
                        st.pyplot(plot_tex_metrics(
                            results['train_tex'], results['train_logn'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_tex'], results['input_logn']
                        ))

                        st.subheader("Best Matches")
                        st.pyplot(plot_best_matches(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['closest_idx_sim'], results['closest_idx_dist'],
                            results['train_filenames'], results['input_logn']
                        ))

                        if results['best_match']:
                            st.subheader("Detailed Peak Comparison")
                            st.pyplot(plot_zoomed_peaks_comparison(
                                results['input_spec'], results['input_freq'],
                                results['best_match']
                            ))

                            #INDIVIDUAL



                            st.subheader("Summary Comparison")
                            st.pyplot(plot_summary_comparison(
                                results['input_freq'], results['input_spec'],
                                results['best_match'], tmp_path
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