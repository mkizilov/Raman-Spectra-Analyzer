# About

This code was made as Texas A&M Grad student project. I'm not very proud of this code, but it might be very useful who wants to study ways of pre-processing and analyzing Raman spectra.

# Installation
## Clone the repository:
```bash
git clone https://github.com/your-username/raman-spectra-analysis.git
cd raman-spectra-analysis
```
## Install the required dependencies:
```bash
pip install pandas numpy matplotlib scipy peakutils scikit-learn
```

# Usage
See example.

# Functions
## Data Loading and Preparation
	•	asc_to_dataframe(filename): Converts a .asc file to a pandas DataFrame.
	•	create_dataframe_list(directory, file_startswith): Creates a list of DataFrames from .asc files in the specified directory.
	•	cut_dataframe(df, start=None, end=None): Cuts the DataFrame to the specified Raman Shift range.
## Plotting functions
	•	plot_raman(df, title="Raman Spectrum"): Plots the Raman spectrum.
## Background removal
	•	remove_background(df_sample, df_background): Removes background from the Raman spectrum.
## Normalization
    •	normalize_spectrum(df, range_min, range_max): Normalizes the Raman spectrum to the specified range.
## Shift to 0
	•	shift_counts_to_zero(df): Shifts the spectrum counts to start from zero.
## Despiking Spectrum
	•	despike_spectrum(df, moving_average, threshold=7, plot=False): Removes spikes from the Raman spectrum.
## Baseline Estimation
    •	estimate_baseline(df, lam=10000000, p=0.05, niter=3, plot=False): Estimates and subtracts the baseline from the Raman spectrum.
## Smoothing Spectrum
	•	smooth_spectrum(df, window_length=11, polyorder=2, plot=False): Smooths the Raman spectrum using the Savitzky-Golay filter.
# Fit Lorentzians
	•	fit_lorentzians(df, n_peaks_to_find=5, title='', remove_peaks_ranges=None, add_peaks=None, threshold=0.25, min_dist=7): Fits Lorentzians to the Raman spectrum.

# License

MIT License

Copyright (c) 2024 Raman Analyzer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.