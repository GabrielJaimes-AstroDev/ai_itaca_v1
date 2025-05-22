import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_pdf import PdfPages
import re
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from matplotlib import rcParams
import tensorflow as tf
import zipfile
from astropy.io import fits

# Configuration settings
rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'medium',
    'axes.labelsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'figure.autolayout': True
})

DEFAULT_CONFIG = {
    'input_spectrum_file': r"C:\1. AI - ITACA\9. JupyterNotebook\_BK\example_4mols_AVER_",
    'output_pdf': "CNN_PEAK_MATCHING_ANALYSIS_71_",
    'trained_models_dir': r"C:\1. AI - ITACA\9. JupyterNotebook\MODELS_01",
    'peak_matching': {
        'sigma_threshold': 1,          
        'sigma_emission': 1.5,         
        'window_size': 3,              
        'fwhm_ghz': 0.08,
        'tolerance_ghz': 0.2,
        'min_peak_height_ratio': 0.5,
        'top_n_lines': 35,
        'debug': True,
        'top_n_similar': 800
    }
}

def safe_float_conversion(value, default=0.0):
    """Safe conversion to float with error handling"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def clean_spectral_data(freq, spec):
    """Remove NaN/inf values and ensure finite data"""
    mask = np.isfinite(freq) & np.isfinite(spec)
    if not np.any(mask):
        raise ValueError("No valid data points remaining after cleaning")
    return freq[mask], spec[mask]

def detect_peaks(x, y, config=DEFAULT_CONFIG):
    """Robust peak detection with error handling"""
    try:
        # Clean input data
        y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Adaptive threshold calculation
        std_dev = np.std(y_clean)
        if std_dev == 0 or not np.isfinite(std_dev):
            std_dev = 1e-10
            
        height_threshold = max(
            config['peak_matching']['sigma_threshold'] * std_dev,
            np.max(y_clean) * 0.01  # Minimum relative threshold
        )
        
        # Find peaks with minimum width constraint
        peaks, properties = find_peaks(
            y_clean, 
            height=height_threshold,
            width=1,
            prominence=height_threshold*0.5
        )
        
        if len(peaks) == 0:
            return []
            
        # Calculate peak widths
        widths = peak_widths(y_clean, peaks, rel_height=0.2)[0] * (x[1] - x[0])
        
        # Create peak dictionary with validation
        valid_peaks = []
        for i, p in enumerate(peaks):
            if (0 <= p < len(x)) and np.isfinite(widths[i]):
                valid_peaks.append({
                    'center': x[p],
                    'height': y_clean[p],
                    'width': widths[i],
                    'left': max(x[p] - widths[i] / 2, x[0]),
                    'right': min(x[p] + widths[i] / 2, x[-1])
                })
        
        return sorted(valid_peaks, key=lambda p: p['height'], reverse=True)
    
    except Exception as e:
        print(f"Peak detection error: {str(e)}")
        return []

def find_input_file(filepath):
    """Find input file with multiple extension attempts"""
    if os.path.exists(filepath):
        return filepath
    
    for ext in ['', '.txt', '.dat', '.fits', '.fit', '.zip']:
        test_path = f"{filepath}{ext}"
        if os.path.exists(test_path):
            return test_path
    
    raise FileNotFoundError(f"File not found: {filepath} (tried extensions: '', '.txt', '.dat', '.fits', '.zip')")

def process_input_file(filepath):
    """Robust file processing with comprehensive error handling"""
    input_logn = None
    input_tex = None
    header = ""
    freq = np.array([])
    spec = np.array([])

    try:
        # Try text formats first
        if filepath.lower().endswith(('.txt', '.dat', '')):
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        lines = [line.strip() for line in file if line.strip()]
                        
                        if not lines:
                            continue
                            
                        header = lines[0]
                        # Extract parameters from header
                        logn_match = re.search(r'logn[\s=:]+([\d.+-]+)', header.lower())
                        tex_match = re.search(r'tex[\s=:]+([\d.+-]+)', header.lower())
                        
                        input_logn = safe_float_conversion(logn_match.group(1)) if logn_match else None
                        input_tex = safe_float_conversion(tex_match.group(1)) if tex_match else None
                        
                        # Process data lines
                        data = []
                        for line in lines[1:]:
                            if line and not line.startswith(("//", "#", "*", "!")):
                                parts = re.split(r'[\s,;|]+', line.strip())
                                if len(parts) >= 2:
                                    try:
                                        freq_val = safe_float_conversion(parts[0]) * 1e9  # Convert to Hz
                                        int_val = safe_float_conversion(parts[1])
                                        if np.isfinite(freq_val) and np.isfinite(int_val):
                                            data.append((freq_val, int_val))
                                    except ValueError:
                                        continue
                        
                        if len(data) >= 10:
                            freq, spec = zip(*data)
                            freq, spec = clean_spectral_data(np.array(freq), np.array(spec))
                            return freq, spec, header, input_logn, input_tex
                except UnicodeDecodeError:
                    continue

        # Try FITS format
        if filepath.lower().endswith(('.fits', '.fit')):
            try:
                with fits.open(filepath) as hdul:
                    if len(hdul) > 1 and 'DATA' in hdul[1].columns.names:
                        table = hdul[1].data
                        all_freqs = []
                        all_intensities = []
                        
                        for row in table:
                            spectrum = row['DATA']
                            crval3 = safe_float_conversion(row.get('CRVAL3', 0))
                            cdelt3 = safe_float_conversion(row.get('CDELT3', 0))
                            crpix3 = safe_float_conversion(row.get('CRPIX3', 1))
                            
                            n = len(spectrum)
                            channels = np.arange(n)
                            frequencies = crval3 + (channels + 1 - crpix3) * cdelt3
                            
                            # Clean and validate
                            valid_mask = np.isfinite(frequencies) & np.isfinite(spectrum)
                            if np.any(valid_mask):
                                all_freqs.append(frequencies[valid_mask])
                                all_intensities.append(spectrum[valid_mask])
                        
                        if all_freqs:
                            combined_freqs = np.concatenate(all_freqs)
                            combined_intensities = np.concatenate(all_intensities)
                            sorted_idx = np.argsort(combined_freqs)
                            return (combined_freqs[sorted_idx], 
                                   combined_intensities[sorted_idx], 
                                   f"FITS: {os.path.basename(filepath)}", 
                                   input_logn, input_tex)
            except Exception as e:
                print(f"FITS processing warning: {str(e)}")

        # Try ZIP archives
        if zipfile.is_zipfile(filepath):
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.lower().endswith('.fits'):
                            with zip_ref.open(zip_info) as fits_file:
                                with fits.open(fits_file) as hdul:
                                    if len(hdul) > 1 and 'DATA' in hdul[1].columns.names:
                                        # Same processing as direct FITS
                                        return process_input_file(fits_file)
            except Exception as e:
                print(f"ZIP processing warning: {str(e)}")

    except Exception as e:
        print(f"Critical file processing error: {str(e)}")
        raise ValueError(f"Failed to process file {filepath}")

    raise ValueError("Unsupported file format or invalid data")

def prepare_input_spectrum(input_freq, input_spec, train_freq, train_data):
    """Prepare input spectrum with range validation and safe interpolation"""
    # Clean input data
    input_freq, input_spec = clean_spectral_data(input_freq, input_spec)
    
    # Validate training data
    train_min_freq = np.min([np.min(f[f > 0]) if len(f) > 0 else np.inf for f in train_freq])
    train_max_freq = np.max([np.max(f) if len(f) > 0 else -np.inf for f in train_freq])
    
    if not np.isfinite(train_min_freq) or not np.isfinite(train_max_freq):
        raise ValueError("Invalid training frequency range")
    
    # Filter input data to training range
    valid_mask = (input_freq >= train_min_freq) & (input_freq <= train_max_freq)
    if not np.any(valid_mask):
        raise ValueError(
            f"Input spectrum range ({input_freq[0]/1e9:.2f}-{input_freq[-1]/1e9:.2f} GHz) "
            f"doesn't overlap with model range ({train_min_freq/1e9:.2f}-{train_max_freq/1e9:.2f} GHz)"
        )
    
    # Safe interpolation
    new_x = np.linspace(train_min_freq, train_max_freq, train_data.shape[1])
    interp_spec = np.interp(
        new_x, 
        input_freq[valid_mask], 
        input_spec[valid_mask], 
        left=0, 
        right=0
    )
    
    return new_x, interp_spec

def match_peaks(input_peaks, synth_peaks, config=DEFAULT_CONFIG):
    """Robust peak matching with validation"""
    used = set()
    matches = []
    
    tolerance_hz = config['peak_matching']['tolerance_ghz'] * 1e9
    min_height_ratio = config['peak_matching']['min_peak_height_ratio']
    
    for ip in input_peaks:
        if 'center' not in ip or 'height' not in ip:
            continue
            
        best = None
        for i, sp in enumerate(synth_peaks):
            if i in used or 'center' not in sp or 'height' not in sp:
                continue
                
            delta_freq = abs(ip['center'] - sp['center'])
            if delta_freq <= tolerance_hz:
                # Normalized intensity difference
                height_diff = abs(ip['height'] - sp['height']) / max(ip['height'], 1e-10)
                
                # Combined score (lower is better)
                score = (
                    0.7 * (delta_freq / tolerance_hz) +
                    0.3 * height_diff
                )
                
                if (best is None or score < best['score']) and (
                    sp['height'] >= ip['height'] * min_height_ratio
                ):
                    best = {
                        'input_peak': ip,
                        'synth_peak': sp,
                        'score': score,
                        'delta_freq': delta_freq,
                        'intensity_diff': height_diff
                    }
        
        if best:
            matches.append(best)
            used.add(synth_peaks.index(best['synth_peak']))
    
    return matches

def calculate_global_score(matches):
    """Calculate weighted global matching score"""
    if not matches:
        return float('inf')
    
    # Weighted average where better matches have more influence
    weights = np.array([1/(m['score'] + 1e-10) for m in matches])
    scores = np.array([m['score'] for m in matches])
    return np.sum(weights * scores) / np.sum(weights)

def enhanced_calculate_metrics(input_spec, train_data, input_peaks=None, train_peaks_list=None, config=DEFAULT_CONFIG):
    """Calculate similarity metrics with enhanced validation"""
    # Input validation
    input_spec = np.nan_to_num(input_spec, nan=0.0)
    train_data = np.nan_to_num(train_data, nan=0.0)
    
    # Cosine similarity with regularization
    input_norm = input_spec / (np.linalg.norm(input_spec) + 1e-10)
    train_norm = train_data / (np.linalg.norm(train_data, axis=1, keepdims=True) + 1e-10)
    similarities = cosine_similarity(input_norm.reshape(1, -1), train_norm.squeeze())[0]
    
    # Peak-based distance metric
    distances = np.zeros(len(train_data))
    
    if input_peaks is None:
        input_peaks = detect_peaks(np.arange(len(input_spec)), input_spec, config)
    
    for i, train_spec in enumerate(train_data):
        train_peaks = train_peaks_list[i] if (train_peaks_list is not None and i < len(train_peaks_list)) else None
        
        if train_peaks is None:
            train_peaks = detect_peaks(np.arange(len(train_spec)), train_spec, config)
        
        # Calculate peak position and intensity differences
        pos_diffs = []
        int_diffs = []
        
        for ip in input_peaks:
            closest_dist = float('inf')
            closest_height_diff = float('inf')
            
            for tp in train_peaks:
                dist = abs(ip['center'] - tp['center'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_height_diff = abs(ip['height'] - tp['height']) / (ip['height'] + 1e-10)
            
            if closest_dist <= config['peak_matching']['window_size'] * 3:
                pos_diffs.append(closest_dist)
                int_diffs.append(closest_height_diff)
        
        if pos_diffs:
            distances[i] = (
                0.5 * np.mean(pos_diffs) +
                0.5 * np.mean(int_diffs)
            )
        else:
            distances[i] = float('inf')
    
    return similarities, distances

# [Previous plotting functions remain exactly the same...]

def analyze_spectrum(filepath, model, train_data, train_freq, train_filenames, train_headers, train_logn, train_tex, config, database_folder):
    """Complete analysis pipeline with comprehensive error handling"""
    try:
        print(f"\nProcessing input spectrum: {filepath}")
        
        # 1. File handling and validation
        filepath = find_input_file(filepath)
        input_freq, input_spec, header, input_logn, input_tex = process_input_file(filepath)
        
        # 2. Data cleaning
        input_freq, input_spec = clean_spectral_data(input_freq, input_spec)
        if len(input_freq) < 10:
            raise ValueError("Insufficient valid data points in input spectrum")
        
        # 3. Prepare input spectrum
        new_x, input_spec = prepare_input_spectrum(input_freq, input_spec, train_freq, train_data)
        
        # 4. Clean training data
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 5. Peak detection
        input_peaks = detect_peaks(new_x, input_spec, config)
        print(f"Detected {len(input_peaks)} peaks in input spectrum")
        
        # 6. Precompute peaks for training spectra
        print("Precomputing peaks for training spectra...")
        train_peaks_list = []
        for i in range(train_data.shape[0]):
            synth_spec = train_data[i,:,0]
            synth_peaks = detect_peaks(train_freq[i], synth_spec, config)
            train_peaks_list.append(synth_peaks[:config['peak_matching']['top_n_lines']])
        
        # 7. Calculate metrics
        similarities, distances = enhanced_calculate_metrics(
            input_spec, train_data.squeeze(), input_peaks, train_peaks_list, config)
        
        # 8. Find top matches
        top_n = min(config['peak_matching']['top_n_similar'], len(train_data))
        top_similar_indices = np.argsort(distances)[:top_n]
        
        # 9. Detailed matching for top candidates
        results = []
        for i in top_similar_indices:
            synth_spec = train_data[i,:,0]
            synth_peaks = train_peaks_list[i]
            
            matches = match_peaks(input_peaks, synth_peaks, config)
            global_score = calculate_global_score(matches)
            
            results.append({
                'index': i,
                'filename': train_filenames[i],
                'header': train_headers[i],
                'x_synth': train_freq[i],
                'y_synth': synth_spec,
                'matches': matches,
                'score': global_score,
                'logn': train_logn[i],
                'tex': train_tex[i],
                'similarity': similarities[i],
                'distance': distances[i],
                'input_logn': input_logn,
                'input_tex': input_tex,
                'input_peaks': input_peaks
            })
        
        # Sort by matching score
        results.sort(key=lambda r: r['score'])
        top_matches = results[:config['peak_matching']['top_n_lines']]
        
        # Find absolute best matches by different metrics
        closest_idx_sim = np.argmax(similarities)
        closest_idx_dist = np.argmin(distances)
        
        best_match = top_matches[0] if top_matches else None
        
        return {
            'input_freq': new_x,
            'input_spec': input_spec,
            'input_peaks': input_peaks,
            'similarities': similarities,
            'distances': distances,
            'top_similar_indices': top_similar_indices,
            'closest_idx_sim': closest_idx_sim,
            'closest_idx_dist': closest_idx_dist,
            'best_match': best_match,
            'top_matches': top_matches,
            'train_logn': train_logn,
            'train_tex': train_tex,
            'train_filenames': train_filenames,
            'input_logn': input_logn,
            'input_tex': input_tex,
            'header': header
        }
        
    except Exception as e:
        print(f"\nERROR during analysis:")
        print(f"File: {filepath}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        raise