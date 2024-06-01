import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
from collections import OrderedDict
import peakutils
from lmfit import Model

# Data Loading and Preparation 
# Use next two function as an example of how to load Raman spectra to dataframe from the file
def load_asc_file(filename):
    """Load .asc file and convert to pandas DataFrame."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Date and Time" in line:
            break
    data = pd.read_csv(filename, sep='\t', skiprows=0, skipfooter=len(lines)-i-1, engine='python')
    data.columns = ['Wavenumber'] + [f'Intensity_{i}' for i in range(1, len(data.columns))]
    data = data.melt(id_vars='Wavenumber', value_name='Intensity')
    return data

def load_data_from_directory(directory, prefix):
    """Load multiple .asc files and create a list of DataFrames."""
    dataframes = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".asc"):
            df = load_asc_file(os.path.join(directory, filename))
            dataframes.append(df)
            filenames.append(filename)
    return dataframes, filenames
def generate_random_raman_spectra(wavenumber_range=(100, 3500), n_points=1000, n_peaks=5, noise_level=0.1, 
                                  peak_positions=None, peak_amplitude_range=(1, 10), peak_width_range=(10, 50), 
                                  baseline_degree=2, baseline_scale=0.0000000001, n_spectra=1, spike_prob=0.001, plot=False):
    """
    Generate random Raman spectra with Lorentzian peaks, polynomial baseline, and Raman spikes.
    
    Parameters:
    wavenumber_range (tuple): The range of wavenumber values (start, end).
    n_points (int): The number of data points in the spectrum.
    n_peaks (int): The number of peaks in the spectrum.
    noise_level (float): The level of Gaussian noise to add to the spectrum.
    peak_positions (list): Specific positions for the peaks. If None, random positions are generated.
    peak_amplitude_range (tuple): The range (min, max) for peak amplitudes.
    peak_width_range (tuple): The range (min, max) for peak widths.
    baseline_degree (int): The degree of the polynomial baseline.
    baseline_scale (float): Scale factor for the polynomial baseline.
    n_spectra (int): Number of spectra to generate.
    spike_prob (float): Probability of a spike occurring at any point.
    plot (bool): Whether to plot the generated spectra.
    
    Returns:
    list of DataFrame: A list of pandas DataFrames containing the wavenumber and intensity values of the generated spectra.
    """
    
    def lorentzian(x, amp, ctr, wid):
        return amp * wid ** 2 / ((x - ctr) ** 2 + wid ** 2)
    
    # Generate wavenumber values
    wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)
    
    # Initialize the spectrum with zeros
    spectrum = np.zeros_like(wavenumbers)
    
    # Generate random peak positions, amplitudes, and widths if not provided
    if peak_positions is None:
        peak_positions = np.random.uniform(wavenumber_range[0], wavenumber_range[1], n_peaks)
    peak_amplitudes = np.random.uniform(peak_amplitude_range[0], peak_amplitude_range[1], n_peaks)
    peak_widths = np.random.uniform(peak_width_range[0], peak_width_range[1], n_peaks)
    
    # Add Lorentzian peaks to the spectrum
    for pos, amp, wid in zip(peak_positions, peak_amplitudes, peak_widths):
        spectrum += lorentzian(wavenumbers, amp, pos, wid)
    
    # Generate a polynomial baseline with scaled coefficients
    baseline_coeffs = np.random.uniform(-1, 1, baseline_degree + 1) * baseline_scale
    baseline = np.polyval(baseline_coeffs, wavenumbers)
    spectrum += baseline
    
    spectra_list = []
    
    for _ in range(n_spectra):
        # Copy the common spectrum (baseline + peaks)
        spectrum_with_noise_and_spikes = spectrum.copy()
        
        # Add Gaussian noise to the spectrum
        noise = np.random.normal(0, noise_level, n_points)
        spectrum_with_noise_and_spikes += noise
        
        # Add random spikes to the spectrum
        spikes = np.random.choice([0, 1], size=n_points, p=[1 - spike_prob, spike_prob])
        spike_magnitudes = np.random.uniform(5, 20, n_points) * spikes
        spectrum_with_noise_and_spikes += spike_magnitudes
        
        # Create a DataFrame with the generated spectrum
        df_spectrum = pd.DataFrame({'Wavenumber': wavenumbers, 'Intensity': spectrum_with_noise_and_spikes})
        spectra_list.append(df_spectrum)
        
        # Plot the generated spectrum if requested
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(wavenumbers, spectrum_with_noise_and_spikes, label=f'Generated Spectrum {_+1}')
            plt.plot(wavenumbers, baseline, label='Polynomial Baseline', linestyle='--')
            plt.xlabel('Wavenumber (cm$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Randomly Generated Raman Spectrum')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    return spectra_list

def average_spectra(dataframes, plot=False):
    """Average a list of DataFrames, ensuring they have the same length and Wavenumber values."""
    min_wavenumber = max(df['Wavenumber'].min() for df in dataframes)
    max_wavenumber = min(df['Wavenumber'].max() for df in dataframes)
    interpolated_dfs = [df.set_index('Wavenumber').reindex(np.linspace(min_wavenumber, max_wavenumber, len(df))) for df in dataframes]
    interpolated_dfs = [df.interpolate(method='linear').reset_index() for df in interpolated_dfs]
    averaged_df = pd.concat(interpolated_dfs).groupby('Wavenumber').mean().reset_index()
    if plot:
        plt.figure(figsize=(10, 6))
        for df in interpolated_dfs:
            plt.plot(df['Wavenumber'], df['Intensity'], alpha=0.5)
        plt.plot(averaged_df['Wavenumber'], averaged_df['Intensity'], color='black', linewidth=2, label='Averaged')
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        plt.title('Averaged Raman Spectrum')
        plt.legend()
        plt.show()
    return averaged_df

def cut_spectra(df, start=None, end=None):
    """Cut the DataFrame to the specified Raman Shift range."""
    if start is None:
        start = df['Wavenumber'].min()
    if end is None:
        end = df['Wavenumber'].max()
    mask = (df['Wavenumber'] >= start) & (df['Wavenumber'] <= end)
    return df.loc[mask].reset_index(drop=True)

# Plotting Functions
def plot_raman_spectrum(df, title="Raman Spectrum"):
    """Plot the Raman spectrum."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Wavenumber'], df['Intensity'])
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Background Removal
def remove_background(df_sample, df_background):
    """Remove background from the Raman spectrum, ensuring same length and Wavenumber values."""
    common_range = np.intersect1d(df_sample['Wavenumber'], df_background['Wavenumber'])
    df_sample = df_sample[df_sample['Wavenumber'].isin(common_range)].sort_values('Wavenumber')
    df_background = df_background[df_background['Wavenumber'].isin(common_range)].sort_values('Wavenumber')
    df_sample['Intensity'] = df_sample['Intensity'] - df_background['Intensity']
    return df_sample

# Normalization
def normalize_spectrum(df, range_min, range_max):
    """Normalize the Raman spectrum to the specified range."""
    df_copy = df.copy()
    mask = (df_copy['Wavenumber'] >= range_min) & (df_copy['Wavenumber'] <= range_max)
    df_copy['Intensity'] = df_copy['Intensity'] * 1000000 / df_copy.loc[mask, 'Intensity'].sum()
    return df_copy

# Shift to Zero
def shift_intensity_to_zero(df):
    """Shift the spectrum intensity to start from zero."""
    df_copy = df.copy()
    df_copy['Intensity'] = df['Intensity'] - df['Intensity'].min()
    return df_copy

# Despiking Spectrum
def despike_spectrum(df, moving_average, threshold=7, plot=False):
    """Remove spikes from the Raman spectrum using modified Z-score."""
    def modified_z_score(y_values):
        y_diff = np.diff(y_values)
        median_y = np.median(y_diff)
        mad_y = np.median([np.abs(y - median_y) for y in y_diff])
        return [0.6745 * (y - median_y) / mad_y for y in y_diff]

    def fix_spikes(y_values, ma, threshold):
        spikes = abs(np.array(modified_z_score(y_values))) > threshold
        y_out = y_values.copy()
        for i in range(len(spikes)):
            if spikes[i]:
                w = np.arange(max(0, i - ma), min(len(y_values) - 1, i + ma))
                we = w[spikes[w] == 0]
                y_out[i] = np.mean(y_values[we])
        return y_out

    def plot_despiking_results(wavelength, original, despiked):
        plt.figure(figsize=(10, 6))
        plt.plot(wavelength, original, label='Original Data')
        plt.plot(wavelength, despiked, label='Despiked Data')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Despiked Spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()

    counts = df['Intensity']
    despiked_counts = fix_spikes(counts, moving_average, threshold)

    df_despiked = df.copy()
    df_despiked['Intensity'] = despiked_counts

    if plot:
        plot_despiking_results(df['Wavenumber'], counts, despiked_counts)
        plt.figure(figsize=(10, 6))
        plt.plot(df['Wavenumber'][1:], modified_z_score(counts), label='Modified Z-score')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Wavenumber')
        plt.ylabel('Z-score')
        plt.title('Modified Z-score for Despiking')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_despiked

# Baseline Estimation
def estimate_baseline(df, lam=10000000, p=0.05, niter=3, plot=False):
    """Estimate and subtract the baseline from the Raman spectrum."""
    def baseline_als(y, lam, p, niter):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    counts = df['Intensity']
    baseline = baseline_als(counts, lam, p, niter)
    baselined_counts = counts - baseline

    df_baselined = df.copy()
    df_baselined['Intensity'] = baselined_counts

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Wavenumber'], counts, label='Original Data')
        plt.plot(df['Wavenumber'], baseline, label='Estimated Baseline', linestyle='--')
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        plt.title('Baseline Estimation')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(df['Wavenumber'], baselined_counts, label='Baselined Data')
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        plt.title('Baselined Spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_baselined

# Smoothing Spectrum
def smooth_spectrum(df, window_length=11, polyorder=2, plot=False):
    """Smooth the Raman spectrum using Savitzky-Golay filter."""
    smoothed_counts = savgol_filter(df['Intensity'], window_length=window_length, polyorder=polyorder)

    df_smoothed = df.copy()
    df_smoothed['Intensity'] = smoothed_counts

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Wavenumber'], df['Intensity'], label='Original Data')
        plt.plot(df['Wavenumber'], smoothed_counts, label='Smoothed Data')
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        plt.title('Smoothed Spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_smoothed

def fit_lorentzians(df, n_peaks_to_find=5, title='', threshold=0.25, min_dist=7, peak_method='scipy', reduce_data_points=None, fixed_centers=True, remove_peaks_ranges=None, add_peaks=None):
    """Fit Lorentzians to the Raman spectrum."""

    def lorentzian(x, amp, ctr, wid):
        return amp * wid ** 2 / ((x - ctr) ** 2 + wid ** 2)

    def func(x, *params):
        y = np.zeros_like(x)
        for i in range(len(params) // 3):
            ctr, amp, wid = params[3 * i], params[3 * i + 1], params[3 * i + 2]
            y += lorentzian(x, amp, ctr, wid)
        return y

    def fit_curve(guess, func, x, y):
        wavenumber_range = max(x) - min(x)
        bounds_lower = [0, 0, 0] * (len(guess) // 3)
        bounds_upper = [np.inf, wavenumber_range / 2, wavenumber_range / 10] * (len(guess) // 3)
        popt, _ = curve_fit(func, x, y, p0=guess, bounds=(bounds_lower, bounds_upper), maxfev=1400000)
        fit = func(x, *popt)
        return popt, fit

    def initial_guess(peaks):
        guess = []
        for peak in peaks:
            width = estimate_peak_width(xs, ys, peak[0])
            guess.extend([peak[0], peak[1], width])
        return guess

    def estimate_peak_width(xs, ys, peak_position):
        half_max = max(ys) / 2.0
        idx = (np.abs(xs - peak_position)).argmin()  # Find the index of the closest value
        left_idxs = np.where(ys[:idx] <= half_max)[0]
        right_idxs = np.where(ys[idx:] <= half_max)[0]
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            width = (max(xs) - min(xs)) / 20  # Default width if half max cannot be found
        else:
            left_idx = left_idxs[-1]
            right_idx = right_idxs[0] + idx
            width = xs[right_idx] - xs[left_idx]
        return width

    def find_peaks_wrapper(xs, ys, n, threshold, min_dist, method):
        if method == 'scipy':
            indexes, _ = find_peaks(ys, height=threshold, distance=min_dist)
        elif method == 'peakutils':
            indexes = peakutils.indexes(ys, thres=threshold, min_dist=min_dist)
        elif method == 'cwt':
            widths = np.arange(1, min_dist)
            indexes = find_peaks_cwt(ys, widths)
        else:
            raise ValueError("Invalid peak finding method.")
        
        if remove_peaks_ranges:
            for r in remove_peaks_ranges:
                indexes = indexes[(xs[indexes] < r[0]) | (xs[indexes] > r[1])]
        
        peaks = np.array([xs[indexes], ys[indexes]]).T
        highest_peaks = sorted(peaks, key=lambda pair: pair[1], reverse=True)[:n]
        
        if add_peaks:
            for p in add_peaks:
                highest_peaks.append([p, ys[(np.abs(xs - p)).argmin()]])
                
        highest_peaks = sorted(highest_peaks, key=lambda pair: pair[0])
        return np.asarray(highest_peaks)

    def plot_fitted_lorentzians(xs, ys, fit, params, peaks, title):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_title(title)
        ax1.plot(xs, ys, lw=1, label='Spectra', color='black')
        ax1.plot(xs, fit, 'r-', label='Fit', color='red', lw=2, ls='--', alpha=0.6)
        for j in range(len(peaks)):
            ctr = peaks[j, 0]
            amp = params[3 * j + 1]
            width = params[3 * j + 2]
            ax1.plot(xs, lorentzian(xs, amp, ctr, width), ls='-', alpha=0.6)
        for peak in peaks:
            ax1.axvline(peak[0], color='grey', linestyle='--', alpha=0.5)
            ax1.text(peak[0] * 0.99, max(ys) * 0.9, f"{int(peak[0])}:{int(peak[1])}", rotation=90, fontsize=8)
        ax1.legend()
        ax1.set_ylabel("Intensity (a.u.)")

        residuals = ys - fit
        ax2.plot(xs, residuals, label='Residuals', color='blue')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
        ax2.set_ylabel("Residuals")
        plt.show()

    xs, ys = df['Wavenumber'], df['Intensity']
    
    if reduce_data_points:
        # Reduce number of points by taking every nth point to improve speed of fitting
        factor = len(xs) // reduce_data_points # Adjust this factor to change the number of points
        xs = xs[::factor]
        ys = ys[::factor]
    
    peaks = find_peaks_wrapper(xs, ys, n_peaks_to_find, threshold, min_dist, peak_method)
    guess = initial_guess(peaks)

    if fixed_centers:
        fixed_centers_values = peaks[:, 0]
        def fixed_func(x, *params):
            y = np.zeros_like(x)
            for i, center in enumerate(fixed_centers_values):
                amp = params[2 * i]
                wid = params[2 * i + 1]
                y += lorentzian(x, amp, center, wid)
            return y
        params, fit = fit_curve(guess, fixed_func, xs, ys)
    else:
        params, fit = fit_curve(guess, func, xs, ys)


    plot_fitted_lorentzians(xs, ys, fit, params, peaks, title)
    return params, fit