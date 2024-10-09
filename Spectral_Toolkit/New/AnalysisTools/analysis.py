import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional, Union
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Assuming you have an 'emission.py' module with 'element_information'
from .emission import element_information


class AnalyticsToolkit:
    """
    A toolkit for handling and analyzing emission spectra data.

    This class provides methods for feature extraction, element identification, and clustering.

    Attributes:
        element_list (List[str]): List of elements to consider in the analysis.
        lower_limit (float): Lower wavelength limit for spectral analysis.
        upper_limit (float): Upper wavelength limit for spectral analysis.
        element_manipulator (element_information): Object for handling element data.
        dbase (Optional[pd.DataFrame]): Processed database of spectral lines.
        features (np.ndarray): Extracted features.
        x_features (List[float]): Wavelengths of extracted features.
    """

    def __init__(self,
                 element_list: List[str] = ["Co", "Si", "K", "O", "Fe", "Rb", "Li", "Mn", "Mg", "P", "Al", "Cu", "Pb", "Cr", "Ti", "C", "Na", "Ca", "Zn", "V"],
                 lower_limit: float = 200,
                 upper_limit: float = 900):
        self.element_list = element_list
        self.element_manipulator = element_information(element_list)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.dbase: Optional[pd.DataFrame] = None
        self.n_clusters = None
        self.model = None
        self.labels = None
        self.n_features = None
        self.x_size = None
        self.y_size = None
        self.fft_metric = None
        self.ft_features = None
        self.int_features = None
        self.features = None
        self.x_features = None

    def automatic_feature_extraction(self, dataset, wavelengths, fft_features: int = 20, intens_features: int = 20,
                                     sigma: float = 0.5, force_recal: bool = False) -> None:
        """
        Automatically extract features from dataset using FFT and intensity methods.

        Args:
            dataset (np.ndarray): The dataset to process.
            wavelengths (np.ndarray): Corresponding wavelengths.
            fft_features (int): Number of FFT features to extract.
            intens_features (int): Number of intensity features to extract.
            sigma (float): Sigma for Gaussian filter.
            force_recal (bool): Force recalculation of FFT metrics.
        """
        if fft_features != 0:
            self.ft_features = self.extract_fft_features(
                dataset=dataset,
                wavelengths=wavelengths,
                n_features=fft_features,
                force_recal=force_recal
            )
        if intens_features != 0:
            self.int_features = self.intensity_features(
                dataset=dataset,
                wavelengths=wavelengths,
                n_features=intens_features
            )
        # Combine features
        tolerance = 0.25
        result_array = self.ft_features.copy() if self.ft_features is not None else np.array([])
        to_check = self.int_features.copy() if self.int_features is not None else np.array([])
        for value in to_check:
            if np.all(np.abs(result_array - value) > tolerance):
                result_array = np.append(result_array, value)
        self.features, self.x_features = self.manual_features(dataset, wavelengths, result_array, sigma=sigma)

    def manual_features(self, dataset, wavelengths, list_of_wavelengths: List[float], sigma: Optional[float] = None):
        """
        Extract the wavelengths provided in list_of_wavelengths from the dataset.

        Args:
            dataset (np.ndarray): The dataset to process.
            wavelengths (np.ndarray): Corresponding wavelengths.
            list_of_wavelengths (List[float]): List of wavelengths to extract.
            sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

        Returns:
            Tuple[np.ndarray, List[float]]: Extracted features and their corresponding wavelengths.
        """
        features = np.array([dataset[:, :, self.wavelength_to_index(wavelengths, wl)] for wl in list_of_wavelengths])
        if sigma is not None:
            features = np.array([gaussian_filter(f, sigma=sigma) for f in features])
        return features, list_of_wavelengths

    def wavelength_to_index(self, wavelengths: np.ndarray, WoI: float) -> int:
        """
        Find index closest to Wavelength of Interest "WoI"

        Args:
            wavelengths (np.ndarray): Array of wavelengths.
            WoI (float): Wavelength of interest.

        Returns:
            int: Index of closest wavelength.
        """
        return np.argmin(np.abs(wavelengths - WoI))

    def extract_fft_features(self, dataset, wavelengths, n_features: int = 20, force_recal: bool = False) -> np.ndarray:
        """
        Extract features using the FFT method.
        """
        if (self.fft_metric is None) or force_recal:
            self.fft_metric = self._fft_features(dataset)
        prominence = np.mean(self.fft_metric) + np.std(self.fft_metric)
        inds, _ = find_peaks(self.fft_metric, distance=3, prominence=prominence)
        fft_wavelengths = wavelengths[inds]
        fft_wavelengths = fft_wavelengths[np.argsort(self.fft_metric[inds])[::-1][:n_features]]
        return fft_wavelengths

    def intensity_features(self, dataset, wavelengths, n_features: int = 20) -> np.ndarray:
        """
        Extract features using the intensity method.
        """
        mean_spec = np.mean(dataset, axis=(0, 1))
        prominence = np.mean(mean_spec) + np.std(mean_spec)
        inds, _ = find_peaks(mean_spec, distance=5, prominence=prominence)
        mean_intensities = mean_spec[inds]
        mean_features = wavelengths[inds]
        mean_features = mean_features[np.argsort(mean_intensities)[::-1][:n_features]]
        return mean_features

    def _fft_features(self, dataset, smallest_dim_pixels: int = 5) -> np.ndarray:
        """
        Spatial Information Ratio algorithm.
        """
        x_size, y_size, _ = dataset.shape
        freqs_x = 2 * np.pi * np.fft.fftfreq(y_size, 0.5)
        freqs_y = 2 * np.pi * np.fft.fftfreq(x_size, 0.5)
        fft_map = np.array([np.fft.fftshift(np.fft.fft2(dataset[:, :, i])) for i in range(dataset.shape[-1])])
        fft_map[:, fft_map.shape[1] // 2, fft_map.shape[2] // 2] = 0  # Remove DC Component

        kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))

        object_size_small = smallest_dim_pixels * 0.5
        size_kspace_small = np.pi / object_size_small

        R = np.sqrt(kxx ** 2 + kyy ** 2)

        sum1 = np.sum(np.abs(fft_map[:, (R < size_kspace_small)]), axis=(1))
        max1 = np.sum(np.abs(fft_map), axis=(1, 2))

        sums = np.divide(sum1, max1, out=np.zeros(sum1.shape, dtype=float))
        sums = np.nan_to_num((sums - np.nanmin(sums)) / (np.nanmax(sums) - np.nanmin(sums)), nan=0.0)

        return sums

    def _classify_intensity(self, value: float) -> str:
        """
        Classify the intensity of a spectral line.

        Args:
            value (float): Relative intensity value.

        Returns:
            str: Classification of the intensity.
        """
        if 0.05 < value < 0.3:
            return "Low Intensity"
        elif 0.3 <= value < 0.5:
            return "Medium Intensity"
        elif 0.5 <= value <= 1:
            return "High Intensity"
        elif 0 <= value < 0.05:
            return "Ultra low Intensity"
        else:
            return "NA"

    def update_elements(self, element_list: List[str]):
        self.element_list = element_list
        self.element_manipulator = element_information(element_list)

    def generate_database(self) -> pd.DataFrame:
        """
        Generate a database of spectral lines for the specified elements.

        Returns:
            pd.DataFrame: DataFrame containing spectral line information.
        """
        element_data = self.element_manipulator.generate_lines_database(max_ion_state=3, lower_limit=self.lower_limit, upper_limit=self.upper_limit)
        header = element_data[0].strip().split(';')
        data_rows = [row.split(';') for row in element_data[1:]]

        df = pd.DataFrame(data_rows, columns=header)
        df['Ion'] = df['Ion'].astype(int)
        df['Line'] = df['Line'].astype(float)
        df['Relative Intensity'] = df['Relative Intensity'].astype(float)
        return df

    def process_database(self, lower_limit: float = 250, upper_limit: float = 800, ion_num: int = 1, min_intensity: float = 0.01) -> np.ndarray:
        """
        Process the spectral line database with specified filters.

        Args:
            lower_limit (float, optional): Lower wavelength limit. Defaults to 250 nm.
            upper_limit (float, optional): Upper wavelength limit. Defaults to 800 nm.
            ion_num (int, optional): Ion state to filter. Defaults to 1.

        Returns:
            np.ndarray: Processed database as a numpy array.
        """
        df = self.generate_database()
        df['Line'] = df['Line'].round(2)
        df = df[(df['Relative Intensity'] > min_intensity) &
                (df['Line'].between(lower_limit, upper_limit)) &
                (df['Ion'] <= ion_num)]

        df['Relative Intensity'] = df['Relative Intensity'].round(3)
        df['Class'] = df['Relative Intensity'].apply(self._classify_intensity)
        df = df.sort_values('Relative Intensity', ascending=False)

        self.dbase = df
        return df.to_numpy()

    def identify_from_elements(self, spectrum_or_cube, wavelengths, operation: str = 'average',
                               prominence: Union[float, str] = 'auto', distance: int = 5,
                               tolerance: float = 0.2, min_intensity: float = 0.5,
                               return_counts: bool = False) -> Dict[str, Union[int, Tuple[int, List[float]]]]:
        """
        Analyze the dataset or spectrum to find peaks and map them to elements in the emission database.

        Args:
            spectrum_or_cube (ndarray): The spectral data cube with shape (x, y, wavelength) or a 1D spectrum array.
            wavelengths (ndarray): Array of corresponding wavelengths.
            operation (str): Operation to perform across the (x, y) spatial dimensions if a spectral cube is provided.
                            Should be 'average' or 'max'.
            prominence (Union[float, str]): The prominence required for peak detection.
                                            If 'auto', it will be set based on the data.
            distance (int): Minimum distance between detected peaks.
            tolerance (float): Tolerance to match peak wavelengths to database entries.
            min_intensity (float): Minimum intensity threshold for emission lines in the database.
            return_counts (bool): If True, return only the element names and their respective counts.

        Returns:
            Dict[str, Union[int, Tuple[int, List[float]]]]:
                If return_counts is False, returns a dictionary with element names as keys, and values are tuples containing:
                - The count of detected lines for the element.
                - A list of the corresponding wavelengths.
                If return_counts is True, returns a dictionary with element names as keys and the count of detected lines as values.
        """

        # If it's a spectral cube, apply the operation to reduce it to a 1D spectrum
        if spectrum_or_cube.ndim == 3:
            if operation == 'average':
                processed_spectrum = np.mean(spectrum_or_cube, axis=(0, 1))
            elif operation == 'max':
                processed_spectrum = np.max(spectrum_or_cube, axis=(0, 1))
            else:
                raise ValueError("Invalid operation. Choose 'average' or 'max'.")
        else:
            # If it's already a 1D spectrum, use it directly
            processed_spectrum = spectrum_or_cube

        # Optimize prominence calculation by precomputing if 'auto'
        if prominence == 'auto':
            prominence = np.mean(processed_spectrum) + np.std(processed_spectrum)

        # Peak detection
        peaks, _ = find_peaks(processed_spectrum, prominence=prominence, distance=distance)
        peak_wavelengths = wavelengths[peaks]

        # Process the emission database once
        emission_database = self.process_database(lower_limit=self.lower_limit,
                                                  upper_limit=self.upper_limit,
                                                  min_intensity=min_intensity)

        # Prepare a more efficient structure for wavelength lookup
        element_wavelengths = np.array([entry[2] for entry in emission_database])
        element_names = [entry[0] for entry in emission_database]
        element_data = {element: (0, []) for element in self.element_list}

        # Optimize peak matching using vectorized operations
        for peak_wavelength in peak_wavelengths:
            differences = np.abs(element_wavelengths - peak_wavelength)
            min_idx = np.argmin(differences)
            if differences[min_idx] < tolerance:
                closest_match = element_names[min_idx]
                count, wavelength_list = element_data[closest_match]
                # Update the tuple in element_data after modifying
                element_data[closest_match] = (count + 1, wavelength_list + [peak_wavelength])

        # Return result based on return_counts flag
        if return_counts:
            return {element: count for element, (count, _) in element_data.items()}

        return {element: (count, wavelength_list) for element, (count, wavelength_list) in element_data.items()}

    def _prepare_data(self, feature_cube):
        """
        Flattens a 3D feature cube into 2D for clustering.
        """
        n_features, x_size, y_size = feature_cube.shape
        self.x_size = x_size
        self.y_size = y_size
        self.n_features = n_features
        return feature_cube.reshape(n_features, -1).T

    def clustering(self, model, n_clusters, feature_cube, scaler=None, **kwargs):
        """
        Applies clustering on a given feature cube using the specified model and number of clusters.

        Args:
            model (str): The clustering model to use ('kmeans' or 'gmm').
            n_clusters (int): Number of clusters to form.
            feature_cube (ndarray): Input feature data in a 3D array (e.g., (n_features, x, y)).
            scaler (str, optional): Scaler type to normalize the data. Supported options are
                                    'minmax' for MinMaxScaler, 'maxabs' for MaxAbsScaler.
                                    If None, no scaling will be applied. Default is None.
            **kwargs: Additional keyword arguments to be passed to the clustering model.

        Returns:
            labels (ndarray): Cluster labels reshaped to match the original feature cube's (x, y) size.
        """
        flattened_data = self._prepare_data(feature_cube=feature_cube)

        if scaler == 'minmax':
            scaler = MinMaxScaler()
            flattened_data = scaler.fit_transform(flattened_data)
        elif scaler == 'maxabs':
            scaler = MaxAbsScaler()
            flattened_data = scaler.fit_transform(flattened_data)
        elif scaler is not None:
            raise ValueError("Unsupported scaler type. Use 'minmax', 'maxabs', or None for no scaling.")

        self.n_clusters = n_clusters

        if model == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, **kwargs)
            self.labels = self.model.fit_predict(flattened_data)
        elif model == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters, **kwargs)
            self.labels = self.model.fit_predict(flattened_data)
        else:
            raise ValueError("Unsupported model type. Use 'kmeans' or 'gmm'.")

        self.x_size, self.y_size = feature_cube.shape[1], feature_cube.shape[2]
        reshaped_labels = self.labels.reshape(self.x_size, self.y_size)

        return reshaped_labels
