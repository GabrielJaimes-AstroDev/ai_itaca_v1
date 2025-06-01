Mira este codigo:

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


rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'medium',
    'axes.labelsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'figure.autolayout': True
})

CONFIG = {
    'input_spectrum_file': r"C:\Users\Usuario\Downloads\1.Examples_molecules\example_2",
    'output_pdf': "CNN_PEAK_MATCHING_ANALYSIS_81",
    'trained_models_dir': 'MODELS_CH3OCHO_narrow6fwhm_3',
    'peak_matching': {
        'sigma_threshold': 1,
        'fwhm_ghz': 0.08,
        'tolerance_ghz': 0.2,
        'min_peak_height_ratio': 0.5,
        'top_n_lines': 35,
        'debug': True,
        'top_n_similar': 800
    }
}

def process_input_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        header = lines[0].strip() if lines else ""
        
        input_logn = None
        input_tex = None
        input_params = re.search(r'logn[\s=:]+([\d.]+).*tex[\s=:]+([\d.]+)', header.lower()) if header else None
        if input_params:
            try:
                input_logn = float(input_params.group(1))
                input_tex = float(input_params.group(2))
            except (ValueError, TypeError):
                input_logn = None
                input_tex = None
        
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith(("//", "#")):
                parts = re.split(r'[\s,;]+', line)
                if len(parts) >= 2:
                    try:
                        freq = float(parts[0]) * 1e9
                        intensity = float(parts[1])
                        data.append((freq, intensity))
                    except ValueError:
                        continue
        
        if len(data) < 10:
            raise ValueError("Insufficient valid data points in spectrum")
        
        freq, spec = zip(*data)
        return np.array(freq), np.array(spec), header, input_logn, input_tex

def prepare_input_spectrum(input_freq, input_spec, train_freq, train_data):
    train_min_freq = np.min(train_freq)
    train_max_freq = np.max(train_freq)
    new_x = np.linspace(train_min_freq, train_max_freq, train_data.shape[1])
    
    valid_mask = (input_freq >= train_min_freq) & (input_freq <= train_max_freq)
    if not np.any(valid_mask):
        raise ValueError("Input spectrum frequency range does not overlap with training data")
    
    interp_spec = np.interp(new_x, input_freq[valid_mask], input_spec[valid_mask], left=0, right=0)
    return new_x, interp_spec.flatten()

def detect_peaks(x, y, sigma=1):
    peaks, _ = find_peaks(y, height=sigma * np.std(y), width=1)
    widths = peak_widths(y, peaks, rel_height=0.2)[0] * (x[1] - x[0])
    return sorted([{
        'center': x[p],
        'height': y[p],
        'width': widths[i],
        'left': x[p] - widths[i] / 2,
        'right': x[p] + widths[i] / 2
    } for i, p in enumerate(peaks)], key=lambda p: p['height'], reverse=True)

def match_peaks(input_peaks, synth_peaks):
    used = set()
    matches = []

    for ip in input_peaks:
        best = None
        for i, sp in enumerate(synth_peaks):
            if i in used:
                continue
            if 'center' not in ip or 'height' not in ip or 'center' not in sp or 'height' not in sp:
                continue
                
            delta_freq = abs(ip['center'] - sp['center'])
            if delta_freq <= CONFIG['peak_matching']['tolerance_ghz'] * 1e9:
                intensity_diff = abs(ip['height'] - sp['height']) / ip['height'] if ip['height'] != 0 else 0
                score = (
                    0.1 * (delta_freq / (CONFIG['peak_matching']['fwhm_ghz'] * 1e9)) +
                    0.9 * intensity_diff
                )
                if not best or score < best['score']:
                    best = {
                        'input_peak': ip,
                        'synth_peak': sp,
                        'score': score,
                        'delta_freq': delta_freq,
                        'intensity_diff': intensity_diff
                    }
        if best:
            matches.append(best)
            used.add(synth_peaks.index(best['synth_peak']))

    return matches

def calculate_global_score(matches):
    if not matches:
        return 999
    return np.mean([m['score'] for m in matches])

def enhanced_calculate_metrics(input_spec, train_data, input_peaks=None, train_peaks_list=None):
    def detect_emission_regions(spec, sigma=1.5, window=3):
        mean = np.mean(spec)
        std = np.std(spec)
        threshold = mean + sigma * std
        peaks, _ = find_peaks(spec, height=threshold)
        return [(max(p - window, 0), min(p + window, len(spec))) for p in peaks]

    def extract_peak_info(spec, regions):
        peaks_info = []
        for start, end in regions:
            sub_spec = spec[start:end]
            if len(sub_spec) == 0:
                continue
            local_idx = np.argmax(sub_spec)
            global_idx = start + local_idx
            height = spec[global_idx]
            peaks_info.append({'center': global_idx, 'height': height})
        return peaks_info

    similarities = cosine_similarity(input_spec.reshape(1, -1), train_data.squeeze())[0]

    distances = np.zeros(len(train_data))

    if input_peaks is None:
        input_regions = detect_emission_regions(input_spec)
        input_peaks = extract_peak_info(input_spec, input_regions)

    for i, train_spec in enumerate(train_data):
        train_peaks = train_peaks_list[i] if train_peaks_list is not None and i < len(train_peaks_list) else None

        if train_peaks is None:
            train_regions = detect_emission_regions(train_spec)
            train_peaks = extract_peak_info(train_spec, train_regions)

        peak_pos_score = 0
        peak_intensity_score = 0
        matched_peaks = 0

        for ip in input_peaks:
            closest_dist = float('inf')
            closest_height_diff = float('inf')

            for tp in train_peaks:
                dist = abs(ip['center'] - tp['center'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_height_diff = abs(ip['height'] - tp['height'])

            if closest_dist <= 3:
                peak_pos_score += closest_dist
                peak_intensity_score += closest_height_diff / ip['height'] if ip['height'] != 0 else 0
                matched_peaks += 1

        if matched_peaks > 0:
            avg_pos_score = peak_pos_score / matched_peaks
            avg_intensity_score = peak_intensity_score / matched_peaks
            distances[i] = avg_pos_score + 2 * avg_intensity_score
        else:
            distances[i] = float('inf')

    return similarities, distances

def plot_similarity_metrics(train_logn, train_tex, similarities, distances, 
                          top_similar_indices, input_logn, input_tex, pdf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sc1 = ax1.scatter(train_logn, similarities, c=train_tex, cmap='viridis', alpha=0.3, label='All spectra')
    ax1.scatter(train_logn[top_similar_indices], similarities[top_similar_indices], 
               c='red', s=30, alpha=0.7, label=f'Top {len(top_similar_indices)} similar')
    if input_logn:
        ax1.axvline(x=input_logn, color='blue', linestyle='--', label=f'Input LogN')
    ax1.set_xlabel('LogN (cm⁻²)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title(f'Top {len(top_similar_indices)} Similar Spectra Selection (by LogN)')
    ax1.legend()
    plt.colorbar(sc1, ax=ax1, label='Tex (K)')
    
    sc2 = ax2.scatter(train_logn, distances, c=train_tex, cmap='viridis', alpha=0.3, label='All spectra')
    ax2.scatter(train_logn[top_similar_indices], distances[top_similar_indices], 
               c='red', s=30, alpha=0.7, label=f'Top {len(top_similar_indices)} similar')
    if input_logn:
        ax2.axvline(x=input_logn, color='blue', linestyle='--', label=f'Input LogN')
    ax2.set_xlabel('LogN (cm⁻²)')
    ax2.set_ylabel('Enhanced Euclidean Distance')
    ax2.legend()
    plt.colorbar(sc2, ax=ax2, label='Tex (K)')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_tex_metrics(train_tex, train_logn, similarities, distances, 
                    top_similar_indices, input_tex, input_logn, pdf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sc1 = ax1.scatter(train_tex, similarities, c=train_logn, cmap='plasma', alpha=0.3, label='All spectra')
    ax1.scatter(train_tex[top_similar_indices], similarities[top_similar_indices], 
               c='red', s=30, alpha=0.7, label=f'Top {len(top_similar_indices)} similar')
    if input_tex:
        ax1.axvline(x=input_tex, color='blue', linestyle='--', label=f'Input Tex')
    ax1.set_xlabel('Tex (K)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title(f'Top {len(top_similar_indices)} Similar Spectra Selection (by Tex)')
    ax1.legend()
    plt.colorbar(sc1, ax=ax1, label='LogN (cm⁻²)')
    
    sc2 = ax2.scatter(train_tex, distances, c=train_logn, cmap='plasma', alpha=0.3, label='All spectra')
    ax2.scatter(train_tex[top_similar_indices], distances[top_similar_indices], 
               c='red', s=30, alpha=0.7, label=f'Top {len(top_similar_indices)} similar')
    if input_tex:
        ax2.axvline(x=input_tex, color='blue', linestyle='--', label=f'Input Tex')
    ax2.set_xlabel('Tex (K)')
    ax2.set_ylabel('Enhanced Euclidean Distance')
    ax2.legend()
    plt.colorbar(sc2, ax=ax2, label='LogN (cm⁻²)')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_best_matches(train_logn, train_tex, similarities, distances, 
                     closest_idx_sim, closest_idx_dist, 
                     train_filenames, input_logn, pdf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sc1 = ax1.scatter(train_logn, similarities, c=train_tex, cmap='viridis', alpha=0.7)
    ax1.scatter(train_logn[closest_idx_sim], similarities[closest_idx_sim], c='red', s=200, 
              marker='*', label=f'Best Match: {train_filenames[closest_idx_sim]}\n(sim={similarities[closest_idx_sim]:.4f})')
    if input_logn:
        ax1.axvline(x=input_logn, color='blue', linestyle='--', label=f'Input LogN')
    ax1.set_xlabel('LogN (cm⁻²)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('LogN vs Similarity')
    ax1.legend()
    plt.colorbar(sc1, ax=ax1, label='Tex (K)')
    
    sc2 = ax2.scatter(train_logn, distances, c=train_tex, cmap='viridis', alpha=0.7)
    ax2.scatter(train_logn[closest_idx_dist], distances[closest_idx_dist], c='red', s=200,
              marker='*', label=f'Best Match: {train_filenames[closest_idx_dist]}\n(dist={distances[closest_idx_dist]:.4f})')
    if input_logn:
        ax2.axvline(x=input_logn, color='blue', linestyle='--', label=f'Input LogN')
    ax2.set_xlabel('LogN (cm⁻²)')
    ax2.set_ylabel('Enhanced Euclidean Distance')
    ax2.set_title('LogN vs Distance')
    ax2.legend()
    plt.colorbar(sc2, ax=ax2, label='Tex (K)')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_individual_matches(new_x, input_spec, match, pdf):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(new_x, input_spec, 'k-', label='Input Spectrum')
    ax1.plot(match['x_synth'], match['y_synth'], 'r-', label='Best Match')
    
    # Verificar si hay matches y si tienen la clave 'center'
    if match['matches']:
        try:
            min_center = min(p.get('center', float('inf')) for p in match['matches'])
            max_center = max(p.get('center', float('-inf')) for p in match['matches'])
            
            if min_center != float('inf') and max_center != float('-inf'):
                x_min = min_center - 1e9
                x_max = max_center + 1e9
                ax1.set_xlim(x_min, x_max)
                
                for m in match['matches']:
                    if 'center' in m and 'height' in m:
                        ip = m['input_peak']
                        sp = m['synth_peak']
                        ax1.axvline(ip['center'], color='blue', linestyle='--', alpha=0.5)
                        ax1.axvline(sp['center'], color='green', linestyle='--', alpha=0.5)
                        ax1.text(ip['center'], ip['height'], f"{ip['center']/1e9:.2f} GHz\n{ip['height']:.1f}K",
                                 color='blue', fontsize=8, ha='center')
                        ax1.text(sp['center'], sp['height'], f"{sp['center']/1e9:.2f} GHz\n{sp['height']:.1f}K",
                                 color='green', fontsize=8, ha='center')
        except KeyError:
            pass
    
    ax1.set_title(f"Match: {match['filename']}\nGlobal Score: {match['score']:.3f} | LogN: {match['logn']:.2f} | Tex: {match['tex']:.2f}")
    ax1.set_ylim(0, max(np.max(input_spec), np.max(match['y_synth'])) * 1.2)
    ax1.legend()
    ax1.grid(alpha=0.2)
    
    y_interp = interp1d(match['x_synth'], match['y_synth'], bounds_error=False, fill_value=0)(new_x)
    ax2.plot(new_x, input_spec - y_interp, color='purple')
    ax2.axhline(0, ls='--', color='gray')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Residual (K)")
    ax2.grid(alpha=0.2)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
def plot_zoomed_peaks_comparison(input_spec, input_freq, best_match, pdf):
    matches = best_match['matches']
    if not matches:
        return
    
    n_matches = len(matches)
    n_cols = min(4, n_matches)
    n_rows = int(np.ceil(n_matches / n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    axs_flat = axs.flatten()
    
    for i, match in enumerate(matches[:n_rows*n_cols]):
        ip = match['input_peak']
        sp = match['synth_peak']
        
        zoom_width = max(ip['width'], sp['width']) * 2
        x_min = min(ip['center'], sp['center']) - zoom_width
        x_max = max(ip['center'], sp['center']) + zoom_width
        
        axs_flat[i].plot(input_freq, input_spec, 'k-', label='Input', linewidth=2)
        axs_flat[i].plot(best_match['x_synth'], best_match['y_synth'], 
                        'r-', label='Best Match', linewidth=2, alpha=0.7)
        
        axs_flat[i].axvline(ip['center'], color='blue', linestyle='--', alpha=0.7)
        axs_flat[i].axvline(sp['center'], color='green', linestyle='--', alpha=0.7)
        
        axs_flat[i].text(ip['center'], ip['height']*1.05, 
                        f"In: {ip['center']/1e9:.4f} GHz\nH: {ip['height']:.2f} K\nW: {ip['width']/1e6:.2f} MHz",
                        color='blue', ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
        
        axs_flat[i].text(sp['center'], sp['height']*1.05, 
                        f"Match: {sp['center']/1e9:.4f} GHz\nH: {sp['height']:.2f} K\nW: {sp['width']/1e6:.2f} MHz",
                        color='green', ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
        
        axs_flat[i].text(0.5, 0.95, 
                        f"ΔFreq: {match['delta_freq']/1e6:.2f} MHz\nScore: {match['score']:.3f}",
                        transform=axs_flat[i].transAxes, ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        axs_flat[i].set_xlim(x_min, x_max)
        axs_flat[i].set_ylim(0, max(ip['height'], sp['height'])*1.2)
        axs_flat[i].set_xlabel("Frequency (GHz)")
        axs_flat[i].set_ylabel("Intensity (K)")
        axs_flat[i].grid(alpha=0.3)
        axs_flat[i].legend(loc='upper right')
        axs_flat[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.4f}'))
    
    for j in range(i+1, len(axs_flat)):
        axs_flat[j].axis('off')
    
    fig.suptitle("Detailed Peak Comparison: Input vs Best Match", y=1.02, fontsize=14)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_summary_comparison(new_x, input_spec, best_match, filepath, pdf):
    fig = plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    ax = fig.add_subplot(gs[0])
    ax.plot(new_x, input_spec, 'k-', label='Input Spectrum')
    ax.plot(best_match['x_synth'], best_match['y_synth'], 'r-', label='Best Match')
    
    for m in best_match['matches']:
        ip = m['input_peak']
        sp = m['synth_peak']
        ax.axvline(ip['center'], color='blue', linestyle='--', alpha=0.3)
        ax.axvline(sp['center'], color='green', linestyle='--', alpha=0.3)
    
    ax.set_title("Best Match Spectrum Comparison", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.2)
    
    text_ax = fig.add_subplot(gs[1])
    text_ax.axis('off')
    
    match_header = best_match['header'][:200] + "..." if len(best_match['header']) > 200 else best_match['header']
    
    info_text = (
        f"INPUT SPECTRUM:\n{'='*50}\n"
        f"• File: {os.path.basename(filepath)}\n"
        f"• LogN: {best_match.get('input_logn', 'Not specified')}\n"
        f"• Tex: {best_match.get('input_tex', 'Not specified')}\n"
        f"• Detected peaks: {len(best_match.get('input_peaks', []))}\n\n"
        f"BEST MATCH (PEAK MATCHING):\n{'='*50}\n"
        f"• File: {best_match['filename']}\n"
        f"• Global score: {best_match['score']:.4f}\n"
        f"• Matched peaks: {len(best_match['matches'])}\n"
        f"• LogN: {best_match['logn']:.2f}\n"
        f"• Tex: {best_match['tex']:.2f}\n"
        f"• Similarity: {best_match['similarity']:.4f}\n"
        f"• Distance: {best_match['distance']:.4f}\n"
        f"• Header: {match_header}"
    )
    
    text_ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='whitesmoke', alpha=0.7), fontfamily='monospace')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def analyze_spectrum(filepath, model, train_data, train_freq, 
                    train_filenames, train_headers, 
                    train_logn, train_tex, pdf, database_folder):
    try:
        print(f"\nProcessing input spectrum: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        input_freq, input_spec, header, input_logn, input_tex = process_input_file(filepath)
        new_x, input_spec = prepare_input_spectrum(input_freq, input_spec, train_freq, train_data)
        
        input_peaks = detect_peaks(new_x, input_spec, CONFIG['peak_matching']['sigma_threshold'])
        input_peaks = input_peaks[:CONFIG['peak_matching']['top_n_lines']]
        
        print("Precomputing peaks for training spectra...")
        train_peaks_list = []
        for i in range(train_data.shape[0]):
            synth_spec = train_data[i,:,0]
            synth_peaks = detect_peaks(train_freq[i], synth_spec)
            train_peaks_list.append(synth_peaks[:CONFIG['peak_matching']['top_n_lines']])
        
        similarities, distances = enhanced_calculate_metrics(
            input_spec, train_data.squeeze(), input_peaks, train_peaks_list)
        
        top_n = CONFIG['peak_matching']['top_n_similar']
        top_similar_indices = np.argsort(distances)[:top_n]
        
        plot_similarity_metrics(train_logn, train_tex, similarities, distances, 
                              top_similar_indices, input_logn, input_tex, pdf)
        
        plot_tex_metrics(train_tex, train_logn, similarities, distances, 
                        top_similar_indices, input_tex, input_logn, pdf)
        
        results = []
        for i in top_similar_indices:
            synth_spec = train_data[i,:,0]
            synth_peaks = train_peaks_list[i]
            
            matches = match_peaks(input_peaks, synth_peaks)
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
        
        results.sort(key=lambda r: r['score'])
        top_matches = results[:CONFIG['peak_matching']['top_n_lines']]
        
        closest_idx_sim = np.argmax(similarities)
        closest_idx_dist = np.argmin(distances)
        
        plot_best_matches(train_logn, train_tex, similarities, distances, 
                         closest_idx_sim, closest_idx_dist, 
                         train_filenames, input_logn, pdf)
        
        for match in top_matches[:5]:
            plot_individual_matches(new_x, input_spec, match, pdf)
        
        best_match = top_matches[0]
        plot_zoomed_peaks_comparison(input_spec, new_x, best_match, pdf)
        plot_summary_comparison(new_x, input_spec, best_match, filepath, pdf)
        
    except Exception as e:
        print(f"Error processing input spectrum: {str(e)}")

def main():
    if not os.path.exists(CONFIG['trained_models_dir']):
        print(f"Error: Directory with trained models not found: {CONFIG['trained_models_dir']}")
        return
    
    model_files = [f for f in os.listdir(CONFIG['trained_models_dir']) 
                  if f.endswith('.keras') and not f.startswith('.')]
    
    if not model_files:
        print("No trained models found in the directory")
        return
    
    if not os.path.exists(CONFIG['input_spectrum_file']):
        print(f"Input spectrum file not found: {CONFIG['input_spectrum_file']}")
        return
    
    for model_file in model_files:
        mol_name = model_file.replace('_model.keras', '')
        pdf_filename = f"{CONFIG['output_pdf']}_{mol_name}.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            try:
                print(f"\nAnalyzing with model: {model_file}")
                
                # Load model
                model_path = os.path.join(CONFIG['trained_models_dir'], model_file)
                model = tf.keras.models.load_model(model_path)
                
                # Load training data
                data_file = os.path.join(CONFIG['trained_models_dir'], f'{mol_name}_train_data.npz')
                if not os.path.exists(data_file):
                    print(f"Training data not found for {mol_name}")
                    continue
                
                with np.load(data_file) as data:
                    train_freq = data['train_freq']
                    train_data = data['train_data']
                    train_logn = data['train_logn']
                    train_tex = data['train_tex']
                    headers = data['headers']
                    filenames = data['filenames']
                
                analyze_spectrum(CONFIG['input_spectrum_file'], model, train_data, train_freq,
                               filenames, headers, train_logn, train_tex, pdf, mol_name)
                
                print(f"Analysis complete for {mol_name}. Results saved to: {pdf_filename}")
                
            except Exception as e:
                print(f"Error analyzing with model {model_file}: {str(e)}")

if __name__ == "__main__":
    main()
