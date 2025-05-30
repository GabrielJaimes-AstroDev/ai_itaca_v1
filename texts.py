PROJECT_DESCRIPTION = """
<div class="description-panel" style="text-align: justify;">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
</div>
"""

PARAMS_EXPLANATION = """
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
"""

TRAINING_DATASET = """
### Training Dataset

| Molécule       | Tex (K)  | LogN (cm⁻²) | Frecuencia (GHz) |
|----------------|----------|-------------|------------------|
| CO             | 20-380   | 12-19.2     | 80-300           |
| SiO            | 20-380   | 12-19.2     | 80-300           |
| HCO⁺           | 20-380   | 12-19.2     | 80-300           |
| CH3CN          | 20-380   | 12-19.2     | 80-300           |
| HNC            | 20-380   | 12-19.2     | 80-300           |
| SO             | 20-380   | 12-19.2     | 80-300           |
| CH3OCHO_Yebes  | 20-350   | 12-19.2     | 20-50            |
| CH3OCHO        | 120-380  | 12-19.2     | 80-300           |
"""

# Main titles
MAIN_TITLE = "AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis"
SUBTITLE = "Molecular Spectrum Analyzer"

# texts.py - Fragmento con FLOW_OF_WORK y ACKNOWLEDGMENTS

# texts.py - Versión optimizada con imágenes después del título

FLOW_OF_WORK = {
    "title": "Flow of Work Diagram",
    "html": """
    <div class="info-panel" style="
        font-family: 'Arial', sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    ">
        <!-- Título centrado -->
        <h3 style="
            text-align: center;
            color: #1E88E5;
            margin-bottom: 5px;
            border-bottom: 2px solid #1E88E5;
            padding-bottom: 10px;
        ">{title}</h3>
        
        <!-- Espacio para imagen (se insertará después) -->
        <div id="workflow-image-container" style="
            margin: 10px 0 20px 0;
            text-align: center;
        "></div>
        
        <!-- Contenido descriptivo -->
        <div style="margin-top: 15px;">
            <h4 style="
                color: #1E88E5; 
                text-align: center;
                margin-bottom: 15px;
            ">Analysis Pipeline</h4>
            
            <ol style="
                padding-left: 25px;
                color: #333;
                line-height: 1.6;
            ">
                <li style="margin-bottom: 10px;"><strong>Spectrum Input:</strong> Upload your observational data file</li>
                <li style="margin-bottom: 10px;"><strong>Pre-processing:</strong> Automatic noise reduction</li>
                <li style="margin-bottom: 10px;"><strong>Peak Detection:</strong> Identify molecular signatures</li>
                <li style="margin-bottom: 10px;"><strong>Model Matching:</strong> Compare with chemical database</li>
                <li style="margin-bottom: 10px;"><strong>Parameter Estimation:</strong> Calculate physical conditions</li>
                <li><strong>Visualization:</strong> Interactive results exploration</li>
            </ol>
        </div>
        
        <div style="
            margin-top: 20px;
            padding: 12px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #1E88E5;
        ">
            <p style="margin: 0; font-size: 0.95em;">
                <strong>Processing Time:</strong> Typically completes in 30-90 seconds depending on spectrum complexity.
            </p>
        </div>
    </div>
    """,
    "image_path": "images/Flow_of_Work.jpg",
    "image_style": """
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0 auto;
    """
}

ACKNOWLEDGMENTS = {
    "title": "Project Acknowledgments",
    "html": """
    <div class="info-panel" style="
        font-family: 'Arial', sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    ">
        <!-- Título centrado -->
        <h3 style="
            text-align: center;
            color: #1E88E5;
            margin-bottom: 5px;
            border-bottom: 2px solid #1E88E5;
            padding-bottom: 10px;
        ">{title}</h3>
        
        <!-- Espacio para imagen (se insertará después) -->
        <div id="acknowledgments-image-container" style="
            margin: 10px 0 20px 0;
            text-align: center;
        "></div>
        
        <!-- Texto de agradecimientos -->
        <div style="
            text-align: justify;
            padding: 0 10px;
            line-height: 1.6;
        ">
            <p>This research is supported by the European Union's <strong>Recovery and Resilience Facility-Next Generation EU</strong>, 
            through the Spanish Government's public business entity Red.es, under the talent attraction and retention program 
            (Investment 4 of Component 19 of the Recovery, Transformation and Resilience Plan).</p>
            
            <p style="margin-top: 10px;">The development of AI-ITACA represents a collaborative effort between astrophysicists, 
            data scientists, and astrochemistry experts to advance our understanding of interstellar chemistry.</p>
        </div>
    </div>
    """,
    "image_path": "images/Acknowledgments.png",
    "image_style": """
        max-width: 80%;
        border-radius: 8px;
        margin: 0 auto;
    """
}
# Cube Visualizer description
CUBE_VISUALIZER_DESCRIPTION = """
<div class="description-panel">
<h3 style="text-align: center; margin-top: 0; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">3D Spectral Cube Visualization</h3>
<p>Upload and visualize ALMA spectral cubes (FITS format) up to 2GB in size. Explore different channels and create integrated intensity maps with these features:</p>
    
<ul style="color: black;">
<li><strong>Interactive Channel Navigation:</strong> Scroll through spectral dimensions</li>
<li><strong>Region Selection:</strong> Extract spectra from specific spatial regions</li>
<li><strong>Dynamic Scaling:</strong> Linear, logarithmic, or square root intensity scaling</li>
<li><strong>Frequency Information:</strong> Automatic detection of spectral axis</li>
</ul>
    
<div class="pro-tip">
<p><strong>Pro Tip:</strong> For best performance with large cubes, select smaller regions when extracting spectra.</p>
</div>
</div>
"""


# Training Dataset (example - include your full content)
TRAINING_DATASET = """
<table class="training-table">
<your full training dataset table here>
</table>
"""
