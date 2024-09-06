import torch
import torch.nn.functional as F

class MultiModalAlignmentMetric(torch.nn.Module):
    def __init__(self, sigma=1.5):
        super().__init__()
        self.sigma = sigma

    def gaussian_kernel_2d(self, kernel_size, sigma):
        """Creates a 2D Gaussian kernel"""
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / (kernel.sum() + 1e-8)

    def compute_gradients(self, image):
        # Create 2D Gaussian kernel for x and y directions
        kernel = self.gaussian_kernel_2d(5, self.sigma).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 5, 5]
        
        # Apply symmetric padding to ensure the same output size
        grad_x = F.conv2d(image, kernel, padding=2)
        grad_y = F.conv2d(image, kernel.transpose(2, 3), padding=2)

        # Debug: Print sizes after convolution to ensure they match
        # print(f"grad_x shape: {grad_x.shape}, grad_y shape: {grad_y.shape}")

        if grad_x.size() != grad_y.size():
            raise RuntimeError(f"Size mismatch: grad_x {grad_x.size()}, grad_y {grad_y.size()}")

        return torch.sqrt(grad_x ** 2 + grad_y ** 2)

    def forward(self, image1, image2):
        # Ensure the images are the same size
        if image1.size() != image2.size():
            image2 = F.interpolate(image2, size=image1.shape[-2:], mode='bilinear', align_corners=False)

        grad1 = self.compute_gradients(image1)
        grad2 = self.compute_gradients(image2)

        # Normalize gradients
        grad1 = grad1 / (grad1.max().clamp(min=1e-8))
        grad2 = grad2 / (grad2.max().clamp(min=1e-8))

        # Compute gradient correlation
        correlation = (grad1 * grad2).mean()

        # Compute structural similarity
        mu1 = F.avg_pool2d(image1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(image2, kernel_size=11, stride=1, padding=5)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(image1 * image1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(image2 * image2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(image1 * image2, kernel_size=11, stride=1, padding=5) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        ssim = ssim_map.mean()

        # Combine metrics
        return correlation * ssim
    
    def mi(self, source, target, **kwargs):
        """
        (Normalized) mutual information

        :param source:
        :param target:
        :param mask:
        :return:
        """
        return self.forward(source, target, **kwargs)
