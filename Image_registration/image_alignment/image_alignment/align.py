import numpy as np
import torch
from .visualize import visualize_affine_transform_with_sliders
from .optimizer import AffineOptimizer
from .affine_transform import AffineTransform
import matplotlib.pyplot as plt

class ImageAligner:
    def __init__(self, image1, image2, device='cpu'):
        self.image1 = image1
        self.image2 = image2
        self.device = device
        self.initial_params = None
        self.final_params = None
        self.optimizer = None

    def manual_align(self):
        """Perform manual alignment using interactive sliders."""
        self.initial_params = visualize_affine_transform_with_sliders(self.image1, self.image2)
        return self.initial_params

    def auto_align(self, optimizer_params=None, transform_params=None):
        """
        Perform automatic alignment using mutual information optimization.
        
        Args:
        optimizer_params (dict): Parameters for the optimizer
        transform_params (dict): Parameters for the affine transform
        """
        if self.initial_params is None:
            raise ValueError("Run manual_align first or provide initial parameters.")

        if optimizer_params is None:
            optimizer_params = {}
        if transform_params is None:
            transform_params = {}

        self.optimizer = AffineOptimizer(
            self.image1, self.image2, 
            initial_params=self.initial_params,
            device=self.device,
            **optimizer_params
        )

        self.final_params = self.optimizer.optimize(**transform_params)
        return self.final_params

    def align(self, optimizer_params=None, transform_params=None):
        """Perform full image alignment process: manual followed by automatic."""
        self.manual_align()
        return self.auto_align(optimizer_params, transform_params)
    
    def align_dataset(self, dataset, y_size):
        """
        Apply the found alignment parameters to a stack of images.

        Parameters:
        - image_array: np.ndarray of shape (x_size, y_size, n_images)
        
        Returns:
        - transformed_images: np.ndarray of shape (x_size, y_size, n_images) with aligned images
        """
        _, _, n_images = dataset.shape

        if self.final_params is None:
            raise ValueError("Alignment parameters (final_params) are not set. Run alignment first.")

        transformed_dataset = np.zeros((y_size[0], y_size[1], dataset.shape[-1]))
        affine_transform = AffineTransform(scale_x_pred = self.final_params['scale_x'],
                                           scale_y_pred = self.final_params['scale_y'],
                                           translation_x_pred = self.final_params['translate'][0],
                                           translation_y_pred = self.final_params['translate'][1],
                                           angle_pred = self.final_params['rot_angle'])
        affine_transform.task = 'nearest'
        for i in range(n_images):
            print(f'Progress: {i+1}/{n_images}', end = '\r')
            image = torch.tensor(dataset[:, :, i]).float().unsqueeze(0).unsqueeze(0)
            transformed_image = affine_transform(image, [1, 1, y_size[0], y_size[1]])
            transformed_dataset[:, :, i] = transformed_image.squeeze(0).squeeze(0).cpu().detach().numpy()
        return transformed_dataset
    
    def plot_alignment(self):
        """
        Plot the initial and final alignments.
        
        Args:
        save_path (str): Path to save the plot. If None, the plot will be displayed instead.
        """
        if self.final_params is None:
            raise ValueError("Run auto_align or align first to get the final parameters.")

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Image Alignment Results")

        # Plot original images
        axs[0, 0].imshow(self.image1, cmap='gray')
        axs[0, 0].set_title("Original Image 1")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(self.image2, cmap='gray')
        axs[0, 1].set_title("Original Image 2")
        axs[0, 1].axis('off')

        # Plot initial alignment
        initial_transform = AffineTransform(
            scale_x_pred=self.initial_params['scale_x'],
            scale_y_pred=self.initial_params['scale_y'],
            translation_x_pred=self.initial_params['translate'][0],
            translation_y_pred=self.initial_params['translate'][1],
            angle_pred=self.initial_params['rot_angle'],
            device=self.device
        )
        image1_tensor = torch.tensor(self.image1).unsqueeze(0).unsqueeze(0).float().to(self.device)
        image2_tensor = torch.tensor(self.image2).unsqueeze(0).unsqueeze(0).float().to(self.device)
        initial_aligned = initial_transform(image1_tensor, image2_tensor.size()).squeeze().cpu().detach().numpy()

        axs[1, 0].imshow(self.image2, cmap='gray', alpha=0.5)
        axs[1, 0].imshow(initial_aligned, cmap='jet', alpha=0.5)
        axs[1, 0].set_title("Initial Alignment")
        axs[1, 0].axis('off')

        # Plot final alignment
        final_transform = AffineTransform(
            scale_x_pred=self.final_params['scale_x'],
            scale_y_pred=self.final_params['scale_y'],
            translation_x_pred=self.final_params['translate'][0],
            translation_y_pred=self.final_params['translate'][1],
            angle_pred=self.final_params['rot_angle'],
            device=self.device
        )
        final_aligned = final_transform(image1_tensor, image2_tensor.size()).squeeze().cpu().detach().numpy()

        axs[1, 1].imshow(self.image2, cmap='gray', alpha=0.5)
        axs[1, 1].imshow(final_aligned, cmap='jet', alpha=0.5)
        axs[1, 1].set_title("Final Alignment")
        axs[1, 1].axis('off')

        plt.tight_layout()