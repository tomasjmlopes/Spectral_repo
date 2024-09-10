import numpy as np
from scipy import stats, signal
from sklearn.metrics import mutual_info_score
from typing import List
from tqdm import tqdm

class ImageSimilarity:
    def __init__(self, reference_image: np.ndarray):
        """
        Initialize the ImageSimilarity class with a reference image.
        
        :param reference_image: 2D numpy array representing the reference image
        """
        self.reference_image = reference_image
        self.metrics = {
            'nmi': self.normalized_mutual_information,
            'sam': self.spectral_angle_mapper,
            'pearson': self.pearson_correlation,
            'spearman': self.spearman_correlation,
            'ssim': self.structural_similarity
        }

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    def normalized_mutual_information(self, image: np.ndarray, bins: int = 256) -> float:
        """Compute normalized mutual information between reference and given image."""
        hist_2d, _, _ = np.histogram2d(self.reference_image.ravel(), image.ravel(), bins=bins)
        mi = mutual_info_score(None, None, contingency=hist_2d)
        entropy1 = stats.entropy(np.histogram(self.reference_image, bins=bins)[0])
        entropy2 = stats.entropy(np.histogram(image, bins=bins)[0])
        return 2 * mi / (entropy1 + entropy2)

    def spectral_angle_mapper(self, image: np.ndarray) -> float:
        """Compute Spectral Angle Mapper (SAM) between reference and given image."""
        dot_product = np.sum(self.reference_image * image, axis=-1)
        norm1 = np.linalg.norm(self.reference_image, axis=-1)
        norm2 = np.linalg.norm(image, axis=-1)
        sam = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
        return np.mean(sam)

    def pearson_correlation(self, image: np.ndarray) -> float:
        """Compute Pearson correlation coefficient between reference and given image."""
        return stats.pearsonr(self.reference_image.ravel(), image.ravel())[0]

    def spearman_correlation(self, image: np.ndarray) -> float:
        """Compute Spearman rank correlation coefficient between reference and given image."""
        return stats.spearmanr(self.reference_image.ravel(), image.ravel())[0]

    def structural_similarity(self, image: np.ndarray, win_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
        """Compute Structural Similarity Index (SSIM) between reference and given image."""
        C1 = (k1 * 255)**2
        C2 = (k2 * 255)**2
        win = np.ones((win_size, win_size)) / (win_size**2)

        mu1 = signal.convolve2d(self.reference_image, win, mode='valid', boundary='symm')
        mu2 = signal.convolve2d(image, win, mode='valid', boundary='symm')

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = signal.convolve2d(self.reference_image**2, win, mode='valid', boundary='symm') - mu1_sq
        sigma2_sq = signal.convolve2d(image**2, win, mode='valid', boundary='symm') - mu2_sq
        sigma12 = signal.convolve2d(self.reference_image * image, win, mode='valid', boundary='symm') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    def compare_image(self, image: np.ndarray, metric: str = None) -> float:
        """
        Compare given image to reference image using specified metric or all metrics.
        
        :param image: Image to compare with the reference image
        :param metric: Specific metric to use. If None, all metrics are computed
        :return: Dictionary of metric results or single metric result
        """
        if metric is not None:
            if metric not in self.metrics:
                raise ValueError(f"Unknown metric: {metric}. Available metrics are: {', '.join(self.metrics.keys())}")
            return self.metrics[metric](image)
        return {name: metric_func(image) for name, metric_func in self.metrics.items()}

    def compare_datacube(self, datacube: np.ndarray, metric: str = None, sort = False) -> List[float]:
        """
        Compare each image in datacube to reference image using specified metric or all metrics.
        
        :param datacube: 3D array where the last dimension represents different images
        :param metric: Specific metric to use. If None, all metrics are computed
        :return: List of comparison results for each image in the datacube
        """
        results = []
        for i in tqdm(range(datacube.shape[-1])):
            image = datacube[..., i]
            comparison = self.compare_image(image, metric)
            results.append(comparison)
        return results