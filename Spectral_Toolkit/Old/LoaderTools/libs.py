import os
import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt

from typing import List, Union, Optional, Tuple
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view

class LibsLoader:
    """
    Python Toolkit designed to handle LIBS datasets.

    This class provides tools for loading data  and processing spectral data (normalization and baseline removal)
    It also inlcudes feature extraction operations, those being automatic extraction, which includes average spectrum 
    and SIR metric analysis, as well as manual extraction for context-based applications

    Attributes:
        fname (str): The filename of the LIBS dataset.
        config (dict): Configuration parameters.
        dataset (np.ndarray): The loaded LIBS dataset.
        wavelengths (np.ndarray): The wavelengths corresponding to the spectral dimension.
        positions (np.ndarray): The positions of each spectrum.
        x_size (int): The size of the x dimension.
        y_size (int): The size of the y dimension.
        spectral_size (int): The size of the spectral dimension.
        features (np.ndarray): Extracted features.
        x_features (List[float]): Wavelengths of extracted features.
    """

    def __init__(self, fname: str, config_file: Optional[str] = None):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"The file {fname} does not exist.")
        self.fname = fname
        self.config = self._load_config(config_file)
        self.dataset = None
        self.wavelengths = None
        self.positions = None
        self.x_size = None
        self.y_size = None
        self.spectral_size = None
        self.fft_metric = None
        self.ft_features = None
        self.int_features = None
        self.features = None
        self.x_features = None

    def _load_config(self, config_file: Optional[str]) -> dict:
        if config_file is None:
            return {'resolution': 0.5}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset(self, init_wv: Optional[int] = None, final_wv: Optional[int] = None, 
                     baseline_corrected: bool = True, return_pos: bool = False) -> None:
        """
        Loads the dataset from the file, optionally focusing on a specified wavelength range.

        Args:
            init_wv (int, optional): Initial wavelength index.
            final_wv (int, optional): Final wavelength index.
            baseline_corrected (bool): If True, applies baseline correction.
            return_pos (bool): If True, loads and stores spatial positions of spectra.
        """
        try:
            with h5py.File(self.fname, 'r') as hf:
                sample = list(hf.keys())[0].split(' ')[-1]
                baseline = 'Pro' if baseline_corrected else "raw_spectrum"

                spectrums = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/{baseline}']) for i in range(len(hf[f'Sample_ID: {sample}']))]
                positions = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/position']) for i in range(len(hf[f'Sample_ID: {sample}']))]

                self.wavelengths = np.array(hf['System properties']['wavelengths']).flatten()
                
                if init_wv is not None and final_wv is not None:
                    spectrums = [s[init_wv:final_wv] for s in spectrums]
                    self.wavelengths = self.wavelengths[init_wv:final_wv]

                self.x_size = len(np.unique([p[1] for p in positions]))
                self.y_size = len(np.unique([p[0] for p in positions]))
                self.spectral_size = len(self.wavelengths)

                # Sort spectrums and positions
                sorted_indices = np.lexsort(([p[0] for p in positions], [p[1] for p in positions]))
                spectrums = [spectrums[i] for i in sorted_indices]
                positions = [positions[i] for i in sorted_indices]

                self.dataset = np.array(spectrums).reshape(self.x_size, self.y_size, self.spectral_size)
                if return_pos:
                    self.positions = np.array(positions)
        except Exception as e:
            raise IOError(f"Error loading dataset: {str(e)}")

    def wavelength_to_index(self, WoI: float) -> int:
        """
        Find index closest to Wavelength of Interest "WoI"

        Args:
            WoI (float): Wavelength of interest

        Returns:
            int: Index of closest wavelength
        """
        return np.argmin(np.abs(self.wavelengths - WoI))

    def normalize_to_sum(self) -> np.ndarray:
        """
        Normalize each spectrum to its sum.

        Returns:
            np.ndarray: The normalized dataset.
        """
        normalized = self.dataset / np.sum(self.dataset, axis=2)[:,:,np.newaxis]
        self.dataset = normalized

    def baseline_correct(self) -> np.ndarray:
        """
        Subtracts the baselines from the spectra.

        Returns:
            np.ndarray: Baseline-corrected dataset
        """
        flat_spectra = self.dataset.reshape(-1, self.spectral_size)
        baselines = self._get_baseline(flat_spectra)
        baselines = baselines[:, :self.spectral_size]  # Align baselines with spectra
        corrected_spectra = (flat_spectra - baselines).reshape(self.x_size, self.y_size, self.spectral_size)
        self.dataset = corrected_spectra

    def manual_features(self, list_of_wavelengths: List[float], sigma: Optional[float] = None) -> np.ndarray:
        """
        Extract the wavelengths provided in list of wavelengths

        Args:
            list_of_wavelengths (List[float]): List of wavelengths to extract
            sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

        Returns:
            np.ndarray: Extracted features
        """
        features = np.array([self.dataset[:, :, self.wavelength_to_index(wl)] for wl in list_of_wavelengths])
        if sigma is not None:
            features = np.array([gaussian_filter(f, sigma=sigma) for f in features])
        self.features = features
        self.x_features = list_of_wavelengths

    def _fft_features(self, dataset, smallest_dim_pixels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spatial information ration algorithm
        """
        freqs_x = 2*np.pi*np.fft.fftfreq(self.y_size, self.config['resolution'])
        freqs_y = 2*np.pi*np.fft.fftfreq(self.x_size, self.config['resolution'])
        fft_map = np.array([np.fft.fftshift(np.fft.fft2(dataset[:, :, i])) for i in range(dataset.shape[-1])])
        fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0  # Remove DC Component
    
        kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))
    
        object_size_small = smallest_dim_pixels * self.config['resolution']
        size_kspace_small = np.pi / object_size_small
    
        R = np.sqrt(kxx**2 + kyy**2)

        sum1 = np.sum(np.abs(fft_map[:, (R < size_kspace_small)]), axis=(1))
        max1 = np.sum(np.abs(fft_map), axis=(1, 2))
    
        sums = np.divide(sum1, max1, out=np.zeros(sum1.shape, dtype=float))
        sums = np.nan_to_num((sums - np.nanmin(sums))/(np.nanmax(sums) - np.nanmin(sums)), nan=0.0)

        return sums

    def extract_fft_features(self, n_features: int = 20, prominence: Union[float, str] = 'auto', 
                             pixel_dist = 5, force_recal = False):
        """
        Automatically extract N features from dataset using the FT Feature Finder.

        Args:
            n_features (int): Number of features to extract
            pixel_dist (int): Smallest pixel dimension
            prominence (float or 'auto'): Prominence for peak finding. If 'auto', it's calculated from the data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and feature metric
        """
        if (self.fft_metric is None) or (force_recal):
            print("Performing the SIR algorithm...")
            self.fft_metric = self._fft_features(dataset = self.dataset, smallest_dim_pixels = pixel_dist)
            print("Operation Completed.")

        if prominence == 'auto':
            prominence = np.mean(self.fft_metric) + np.std(self.fft_metric)

        inds, _ = find_peaks(self.fft_metric, distance=3, prominence=prominence)

        fft_wavelengths = self.wavelengths[inds]
        fft_wavelengths = fft_wavelengths[np.argsort(self.fft_metric[inds])[::-1][:n_features]]
        self.ft_features = fft_wavelengths

    def intensity_features(self, n_features: int = 20, prominence: Union[float, str] = 'auto'):
        """
        Automatically extract N features from dataset using highest peaks from the mean spectrum.

        Args:
            n_features (int): Number of features to extract
            pixel_dist (int): Smallest pixel dimension
            prominence (float or 'auto'): Prominence for peak finding. If 'auto', it's calculated from the data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and feature metric
        """
        mean_spec = self._calculate_average_spectrum(self.dataset)
        if prominence == 'auto':
            prominence = np.mean(mean_spec) + np.std(mean_spec)
        inds, _ = find_peaks(mean_spec, distance = 5, prominence = prominence)

        mean_intensities = mean_spec[inds]
        mean_features = self.wavelengths[inds]
        mean_features = mean_features[np.argsort(mean_intensities)[::-1][:n_features]]
        self.int_features = mean_features

    def automatic_feature_extraction(self, fft_features: int = 20, fft_prominence: float = 'auto', min_pixel_dist: float = 5, 
                                    intens_features: int = 20, int_prominence: float = 'auto', sigma: float = 0.5,
                                    force_recal = False):

        if fft_features != 0:
            self.extract_fft_features(n_features = fft_features, prominence = fft_prominence, 
                                      pixel_dist = min_pixel_dist, force_recal = force_recal)

        if (intens_features != 0) & (self.int_features is None):
            self.intensity_features(n_features = intens_features, prominence = int_prominence)

        tolerance = 0.25
        result_array = self.ft_features.copy() if not(self.fft_metric is None) else np.array([])
        to_check = self.int_features.copy() if not(self.int_features is None) else np.array([])
        for value in to_check:
            if np.all(np.abs(result_array - value) > tolerance):
                result_array = np.append(result_array, value)

        self.manual_features(result_array, sigma = sigma)

    def _get_baseline(self, dataset: np.ndarray, min_window_size: int = 50, smooth_window_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate baseline using rolling window method.

        Args:
            dataset (np.ndarray): Input dataset
            min_window_size (int): Minimum window size for rolling minimum
            smooth_window_size (int, optional): Window size for smoothing

        Returns:
            np.ndarray: Calculated baselines
        """
        if smooth_window_size is None:
            smooth_window_size = 2 * min_window_size

        local_minima = self._rolling_min(
            arr = np.hstack(
                [dataset[:, 0][:, np.newaxis]] *
                ((min_window_size + smooth_window_size) // 2)
                + [dataset]
                + [dataset[:, -1][:, np.newaxis]] *
                ((min_window_size + smooth_window_size) // 2)
            ),
            window_width = min_window_size
        )
        return np.apply_along_axis(arr = local_minima, func1d = np.convolve, axis = 1,
                                   v = self._get_smoothing_kernel(smooth_window_size), mode = 'valid')

    @staticmethod
    def _rolling_min(arr: np.ndarray, window_width: int) -> np.ndarray:
        """
        Calculates the moving minima in each row of the provided array.

        Args:
            arr (np.ndarray): Input array
            window_width (int): Width of the rolling window

        Returns:
            np.ndarray: Array of rolling minimums
        """
        window = sliding_window_view(arr, (window_width,), axis = len(arr.shape) - 1)
        return np.amin(window, axis = len(arr.shape))

    @staticmethod
    def _get_smoothing_kernel(window_width: int) -> np.ndarray:
        """
        Generates a Gaussian smoothing kernel of the desired width.

        Args:
            window_width (int): Width of the smoothing window

        Returns:
            np.ndarray: Gaussian smoothing kernel
        """
        kernel = np.arange(-window_width//2, window_width//2 + 1, 1)
        sigma = window_width // 4
        kernel = np.exp(-(kernel ** 2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def plot_spectrum(self, x: int, y: int) -> None:
        """
        Plot the spectrum at the given x, y coordinates.

        Args:
            x (int): x-coordinate
            y (int): y-coordinate
        """
        spectrum = self.dataset[x, y, :]
        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spectrum)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(f'Spectrum at position ({x}, {y})')
        plt.show()

    def _calculate_average_spectrum(self, dataset) -> np.ndarray:
        """
        Calculate the average spectrum of the whole dataset.

        Returns:
            np.ndarray: The average spectrum
        """
        return np.mean(dataset, axis = (0, 1))