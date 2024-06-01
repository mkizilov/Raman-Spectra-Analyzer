import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from collections import OrderedDict
import peakutils

# Data Loading and Preparation
def asc_to_dataframe(filename):
    """Convert .asc file to pandas DataFrame."""
    start_index = 0
    end_index = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if "Date and Time" in line:
                end_index = i
                break

    data = pd.read_csv(filename, sep='\t', skiprows=start_index, skipfooter=open(filename).read().count('\n') - end_index, engine='python')
    data = data.rename(columns={data.columns[0]: 'Wavenumber', data.columns[1]: 'Intensity'})
    return pd.DataFrame(data)

def create_dataframe_list(directory, file_startswith):
    """Create a list of DataFrames from .asc files in the specified directory."""
    dataframe_list = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.startswith(file_startswith) and filename.endswith(".asc"):
            df = asc_to_dataframe(os.path.join(directory, filename))
            df = cut_dataframe(df, 0, len(df))
            dataframe_list.append(df)
            filenames.append(filename)
    return dataframe_list, filenames

def average_dataframes(dataframe_list):
    """Average the list of DataFrames."""
    
    #
    df = pd.concat(dataframe_list).groupby(level=0).mean()
    return df

def cut_dataframe(df, start=None, end=None):
    """Cut the DataFrame to the specified Raman Shift range."""
    if start is None:
        start = 0
    if end is None:
        end = len(df)
    mask = (df['Wavenumber'] >= start) & (df['Wavenumber'] <= end)
    return df.loc[mask].reset_index(drop=True)

# Plotting Functions
def plot_raman(df, title="Raman spectra"):
    """Plot the Raman spectra."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Wavenumber'], df['Intensity'])
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Background Removal
def remove_background(df_sample, df_background):
    """Remove background from the Raman spectra."""
    df_sample_copy = df_sample.copy()
    df_sample_copy['Intensity'] = df_sample['Intensity'] - df_background['Intensity']
    return df_sample_copy

# Normalization
def normalize_spectra(df, range_min, range_max):
    """Normalize the Raman spectra to the specified range."""
    df_copy = df.copy()
    mask = (df_copy['Wavenumber'] >= range_min) & (df_copy['Wavenumber'] <= range_max)
    df_copy['Intensity'] = df_copy['Intensity'] * 1000000 / df_copy.loc[mask, 'Intensity'].sum()
    return df_copy

# Shift to Zero
def shift_counts_to_zero(df):
    """Shift the spectra counts to start from zero."""
    df_copy = df.copy()
    df_copy['Intensity'] = df['Intensity'] - min(df['Intensity'])
    return df_copy

# Despiking spectra
def despike_spectra(df, moving_average, threshold=7, plot=False):
    """Remove spikes from the Raman spectra."""
    def modified_z_score(y_values):
        y_diff = np.diff(y_values)
        median_y = np.median(y_diff)
        mad_y = np.median([np.abs(y - median_y) for y in y_diff])
        return [0.6745 * (y - median_y) / mad_y for y in y_diff]

    def fix_spikes(y_values, ma, threshold):
        spikes = abs(np.array(modified_z_score(y_values))) > threshold
        y_out = y_values.copy()
        for i in range(len(spikes)):
            if spikes[i] != 0:
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
        plt.title('Despiked spectra')
        plt.legend()
        plt.grid(True)
        plt.show()

    counts = df['Intensity']
    despiked_counts = fix_spikes(counts, moving_average, threshold)

    df_despiked = df.copy()
    df_despiked['Intensity'] = despiked_counts

    if plot:
        plot_despiking_results(df['Wavenumber'], counts, despiked_counts)

    return df_despiked

# Baseline Estimation
def estimate_baseline(df, lam=10000000, p=0.05, niter=3, plot=False):
    """Estimate and subtract the baseline from the Raman spectra."""
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
        plt.title('Baselined spectra')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_baselined

# Smoothing spectra
def smooth_spectra(df, window_length=11, polyorder=2, plot=False):
    """Smooth the Raman spectra using Savitzky-Golay filter."""
    smoothed_counts = savgol_filter(df['Intensity'], window_length=window_length, polyorder=polyorder)

    df_smoothed = df.copy()
    df_smoothed['Intensity'] = smoothed_counts

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Wavenumber'], df['Intensity'], label='Original Data')
        plt.plot(df['Wavenumber'], smoothed_counts, label='Smoothed Data')
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        plt.title('Smoothed spectra')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_smoothed

# Fit Lorentzians
def fit_lorentzians(df, n_peaks_to_find=5, title='', remove_peaks_ranges=None, add_peaks=None, threshold=0.25, min_dist=7):
    """Fit Lorentzians to the Raman spectra."""
    
    def lorentzian(x, amp, ctr, wid):
        return amp * wid ** 2 / ((x - ctr) ** 2 + wid ** 2)

    def func(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr, amp, wid = params[i], params[i + 1], params[i + 2]
            y += lorentzian(x, amp, ctr, wid)
        return y

    def fit_curve(guess, func, x, y):
        sigma = [2] * len(y)
        bounds_lower = [0] * len(guess)
        bounds_upper = [np.inf] * len(guess)
        popt, _ = scipy.optimize.curve_fit(func, x, y, p0=guess, maxfev=1400000, sigma=sigma, bounds=(bounds_lower, bounds_upper))
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
            width = (max(xs) - min(xs)) / 10  # Default width if half max cannot be found
        else:
            left_idx = left_idxs[-1]
            right_idx = right_idxs[0] + idx
            width = xs[right_idx] - xs[left_idx]
        return width

    def find_peaks(xs, ys, n, threshold, min_dist):
        indexes = peakutils.indexes(ys, thres=threshold, min_dist=min_dist)
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
        for j in range(0, len(params), 3):
            ctr, amp, width = params[j], params[j + 1], params[j + 2]
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
    peaks = find_peaks(xs, ys, n_peaks_to_find, threshold, min_dist)
    guess = initial_guess(peaks)

    # Ensure initial guesses are within bounds
    bounds_lower = [0] * len(guess)
    bounds_upper = [np.inf] * len(guess)
    guess = np.clip(guess, bounds_lower, bounds_upper)

    params, fit = fit_curve(guess, func, xs, ys)
    plot_fitted_lorentzians(xs, ys, fit, params, peaks, title)
    return params, fit