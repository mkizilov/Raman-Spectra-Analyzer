# About
This code was made as Texas A&M Grad student project. I'm not very proud of this code, but it might be very useful who wants to study ways of pre-processing and analyzing Raman spectra.

# Overview
The Raman Spectra Analyzer is a Python-based tool designed for processing and analyzing Raman spectra. This tool allows users to generate synthetic Raman spectra, load experimental data, preprocess the data, and fit Lorentzian peaks to the spectra. It includes functions for background removal, normalization, despiking, baseline estimation, smoothing, and peak fitting.

# Features
	• Synthetic Spectra Generation: Generate random Raman spectra with customizable parameters, including Lorentzian peaks, polynomial baseline, Gaussian noise, and random spikes.
	• Data Loading: Load Raman spectra from .asc files into pandas DataFrames.
	• Preprocessing:
	• Cut spectra to specified Raman Shift ranges.
	• Average multiple spectra, ensuring they have the same length and wavenumber values.
	• Remove background spectra.
	• Normalize spectra to a specified range.
	• Shift intensity values to start from zero.
	• Despike spectra using modified Z-scores.
	• Estimate and subtract baselines using adaptive least squares.
	• Smooth spectra using the Savitzky-Golay filter.
	• Peak Fitting: Fit Lorentzian peaks to Raman spectra using different peak finding methods and customizable parameters.
	• Visualization: Plot spectra at various stages of processing for visualization and analysis.


# Installation
## Clone the repository:
```bash
git clone https://github.com/mkizilov/RamanSpectrumAnalyzer.git
cd RamanSpectrumAnalyzer
```
## Install the required dependencies:
To use the Raman Spectra Analyzer, you need to have Python installed along with the following libraries:

	• pandas
	• numpy
	• matplotlib
	• scipy
	• peakutils
	• lmfit
You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib scipy peakutils lmfit
```

# Usage
For an example of how to use the Raman Spectra Analyzer, please refer to the example.ipynb notebook included in this repository. The notebook demonstrates how to generate synthetic Raman spectra, preprocess the data, and fit Lorentzian peaks, with visualizations at each step.  

## Data Loading and Preparation
To use your own data parse it and create a dataframe or list of dataframes. Each dataframe should have colums: Intensity and Wavenumber.
load_asc_file(filename): Load a .asc file and convert it to a pandas DataFrame.  
load_data_from_directory(directory, prefix): Load multiple .asc files from a directory and create a list of DataFrames.  
## Synthetic Spectra Generation
generate_random_raman_spectra(...): Generate random Raman spectra with Lorentzian peaks, polynomial baseline, Gaussian noise, and random spikes.  

## Preprocessing
cut_spectra(df, start=None, end=None): Cut the DataFrame to the specified Raman Shift range.  
average_spectrum(dataframes, plot=False): Average a list of DataFrames, ensuring they have the same length and wavenumber values.  
remove_background(df_sample, df_background): Remove background from the Raman spectra, ensuring same length and wavenumber values.  
normalize_spectra(df, range_min, range_max): Normalize the Raman spectra to the specified range.  
shift_intensity_to_zero(df): Shift the spectra intensity to start from zero.  
despike_spectra(df, moving_average, threshold=7, plot=False): Remove spikes from the Raman spectra using modified Z-score.  
estimate_baseline(df, lam=10000000, p=0.05, niter=3, plot=False): Estimate and subtract the baseline from the Raman spectra.  
smooth_spectra(df, window_length=11, polyorder=2, plot=False): Smooth the Raman spectra using the Savitzky-Golay filter.  

## Peak Fitting
fit_lorentzians(df, n_peaks_to_find=5, title='', threshold=0.25, min_dist=7, peak_method='scipy', reduce_data_points=None, fixed_centers=True, remove_peaks_ranges=None, add_peaks=None): Fit Lorentzians to the Raman spectra.  

## Plotting
plot_raman_spectra(df, title="Raman spectra"): Plot the Raman spectra.  

# Contribution
Contributions to the Raman Spectra Analyzer are welcome! If you have any bug reports, feature requests, or code improvements, please open an issue or submit a pull request.  

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