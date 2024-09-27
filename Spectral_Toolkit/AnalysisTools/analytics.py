import numpy as np
import pandas as pd

from .emission import element_information
from typing import Dict, List, Tuple, Optional, Union
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class AnalyticsToolkit:
    """
    A toolkit for handling and analyzing emission spectra data.

    This class provides methods for loading element data, identifying spectral lines,
    and classifying emission intensities.

    Attributes:
        element_list (List[str]): List of elements to consider in the analysis.
        lower_limit (float): Lower wavelength limit for spectral analysis.
        upper_limit (float): Upper wavelength limit for spectral analysis.
        element_manipulator (element_information): Object for handling element data.
        dbase (Optional[pd.DataFrame]): Processed database of spectral lines.
        id (Optional[Dict[str, List[Tuple[float, float, float, str]]]]): Identified elements and their spectral lines.
    """

    def __init__(self, 
                 element_list: List[str] = ["Co", "Si", "K", "O", "Fe", "Rb", "Li", "Mn", "Mg", "P", "Al", "Cu", "Pb", "Cr", "Ti", "C", "Na", "Ca", "Zn", "V"],
                 lower_limit: float = 200,
                 upper_limit: float = 900):
        """
        Initialize the EmissionToolkit.

        Args:
            element_list (List[str], optional): List of elements to consider. Defaults to a predefined list.
            lower_limit (float, optional): Lower wavelength limit. Defaults to 200 nm.
            upper_limit (float, optional): Upper wavelength limit. Defaults to 900 nm.
        """
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

    @staticmethod
    def _classify_intensity(value: float) -> str:
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
        
    def update_elements(self, element_list: list[str]):
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

    def identify_from_features(self, x_features: List[float], wavelength_tolerance: float = 0.5, 
                          ion_num: int = 2, min_intensity: float = 0.01, return_counts: bool = False) -> Dict[str, Union[List[Tuple[float, float, float, str]], int]]:
        """
        Identify elements based on the emission lines present in x_features.

        Args:
            x_features (List[float]): List of feature wavelengths to analyze.
            wavelength_tolerance (float, optional): Wavelength tolerance for matching. Defaults to 0.5 nm.
            ion_num (int, optional): Ionization number to filter elements. Defaults to 2.
            min_intensity (float, optional): Minimum intensity threshold for lines. Defaults to 0.01.
            return_counts (bool, optional): If True, return the number of peaks found per element instead of matched lines. Defaults to False.

        Returns:
            Dict[str, Union[List[Tuple[float, float, float, str]], int]]:
                If return_counts is False (default), the dictionary contains element names as keys and lists of tuples for each match.
                Each tuple contains (feature, matched wavelength, intensity, category).
                
                If return_counts is True, the dictionary contains element names as keys and the number of matched peaks as values.
        """
        # Get the emission lines from the database, filtered by min_intensity and ion_num
        emission_lines = self.process_database(lower_limit = self.lower_limit, 
                                               upper_limit = self.upper_limit, 
                                               min_intensity=min_intensity, 
                                               ion_num=ion_num)
        identified_elements = {}

        for feature in x_features:
            closest_match = None
            min_difference = float('inf')

            for element, _, wavelength, intensity, category in emission_lines:
                difference = abs(feature - wavelength)
                
                if difference <= wavelength_tolerance and difference < min_difference:
                    closest_match = (element, feature, wavelength, intensity, category)
                    min_difference = difference
            
            if closest_match:
                element, feature, wavelength, intensity, category = closest_match
                if element not in identified_elements:
                    identified_elements[element] = [] if not return_counts else 0
                
                if return_counts:
                    identified_elements[element] += 1
                else:
                    identified_elements[element].append((feature, wavelength, intensity, category))
        
        return identified_elements


    def identify_from_elements(self, spectral_cube, wavelengths, operation: str = 'average', prominence: Union[float, str] = 'auto', 
                               distance: int = 5, tolerance: float = 0.2, min_intensity: float = 0.5, return_counts: bool = False) -> Dict[str, Union[int, Tuple[int, List[float]]]]:
        """
        Analyze the dataset to find peaks and map them to elements in the emission database.

        Args:
            spectral_cube (ndarray): The spectral data cube with shape (x, y, wavelength).
            wavelengths (ndarray): Array of corresponding wavelengths.
            operation (str): Operation to perform across the (x, y) spatial dimensions. 
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

        if operation == 'average':
            if len(spectral_cube.shape) == 3:
                processed_spectrum = np.mean(spectral_cube, axis=(0, 1))
            elif len(spectral_cube.shape) == 2:
                processed_spectrum = np.mean(spectral_cube, axis=(0))
            else:
                raise ValueError("Invalid spectral cube shape")
        elif operation == 'max':
            if len(spectral_cube.shape) == 3:
                processed_spectrum = np.max(spectral_cube, axis=(0, 1))
            elif len(spectral_cube).shape == 2:
                processed_spectrum = np.max(spectral_cube, axis=(0))
            else:
                raise ValueError("Invalid spectral cube shape")
        else:
            raise ValueError("Invalid operation. Choose 'average' or 'max'.")

        if prominence == 'auto':
            prominence = np.mean(processed_spectrum) + np.std(processed_spectrum)

        peaks, _ = find_peaks(processed_spectrum, prominence=prominence, distance=distance)
        peak_wavelengths = wavelengths[peaks]

        emission_database = self.process_database(lower_limit= self.lower_limit, 
                                                  upper_limit = self.upper_limit,
                                                  min_intensity=min_intensity)
        element_data = {element_entry[0]: (0, []) for element_entry in emission_database}

        for peak_wavelength in peak_wavelengths:
            closest_match = None
            min_difference = float('inf')

            for element_entry in emission_database:
                element, _, line_wavelength, _, _ = element_entry
                difference = abs(peak_wavelength - line_wavelength)
                
                if difference < tolerance and difference < min_difference:
                    closest_match = element
                    min_difference = difference
            
            if closest_match:
                count, wavelength_list = element_data[closest_match]
                element_data[closest_match] = (count + 1, wavelength_list + [peak_wavelength])

        if return_counts:
            return {element: count for element, (count, _) in element_data.items()}

        return {element: (count, wavelength_list) for element, (count, wavelength_list) in element_data.items()}
    
    def identify_on_cluster(self, cluster_number: int, spectral_cube, *args, **kwargs):
        """
        Applies a given operation (like 'identify_from_elements') to only the data within a specific cluster.

        Args:
            cluster_number (int): The cluster number on which the operation should be performed.
            spectral_cube (ndarray): The original 3D spectral cube (x, y, wavelength).
            operation_fn (callable): The operation function to be applied (e.g., identify_from_elements).
            *args: Additional positional arguments to pass to the operation function.
            **kwargs: Additional keyword arguments to pass to the operation function.

        Returns:
            The result of the operation function applied to the specified cluster.
        """
        if self.labels is None:
            raise ValueError("Clustering has not been performed yet. Please cluster the data first.")

        mask = self.labels == cluster_number
        mask = mask.reshape(self.x_size, self.y_size)

        filtered_cube = spectral_cube[mask]

        filtered_cube = filtered_cube.reshape(-1, spectral_cube.shape[-1])
        result = self.identify_from_elements(filtered_cube, *args, **kwargs)

        return result

    def _prepare_data(self, feature_cube):
        """
        Flattens a 3D feature cube into 2D for clustering.
        For example, from (z, x, y) to (z, x*y).
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
                feature_cube (ndarray): Input feature data in a 3D array (e.g., (x, y, z)).
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

            reshaped_labels = self.labels.reshape(self.x_size, self.y_size)

            return reshaped_labels

    