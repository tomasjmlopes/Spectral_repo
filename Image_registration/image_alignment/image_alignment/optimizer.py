import torch
from .affine_transform import AffineTransform
from .mi import MI
import numpy as np

class AffineOptimizer:
    def __init__(self, image1, image2, initial_params, device='cpu', 
                 optim='SGD', lr=1e-4, momentum=0.85, num_iters=400):
        self.image1 = torch.tensor(image1).unsqueeze(0).unsqueeze(0).float().to(device)
        self.image2 = torch.tensor(image2).unsqueeze(0).unsqueeze(0).float().to(device)
        self.initial_params = initial_params
        self.device = device
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.num_iters = num_iters
        self.mutual_info = MI(dimension=2, num_bins=32, kernel_sigma=1).to(device)

    def optimize(self, s_tol=0.2, t_tol=0.3, a_tol=np.pi/6, edge_align=False):
        if edge_align:
            self.image1 = self._edge_detection(self.image1)
            self.image2 = self._edge_detection(self.image2)

        model = AffineTransform(
            scale_x_pred=self.initial_params['scale_x'],
            scale_y_pred=self.initial_params['scale_y'],
            translation_x_pred=self.initial_params['translate'][0],
            translation_y_pred=self.initial_params['translate'][1],
            angle_pred=self.initial_params['rot_angle'],
            s_tol=s_tol, t_tol=t_tol, a_tol=a_tol,
            device=self.device
        )

        if self.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, maximize = True)
        elif self.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, maximize=True)
        else:
            raise ValueError('Optimizer not supported. Choose between Adam and SGD')

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.85, end_factor=1, total_iters=self.num_iters)

        best_params = None
        best_loss = float('-inf')

        for epoch in range(self.num_iters):
            optimizer.zero_grad()
            transformed_image = model(self.image1, self.image2.size())
            loss = self.mutual_info.mi(transformed_image, self.image2, mask=transformed_image != 0)

            print(f'Epoch {epoch}/{self.num_iters}, Current Loss: {loss:.6f}, Best Loss: {best_loss:.3f}', end = '\r')

            if loss.item() > best_loss:
                best_loss = loss.item()
                best_params = {
                    'rot_angle': model.angle.item(),
                    'translate': [model.transX.item(), model.transY.item()],
                    'scale_x': model.scalex.item(),
                    'scale_y': model.scaley.item(),
                }
            loss.backward()
            optimizer.step()
            scheduler.step()

        return best_params

    @staticmethod
    def _edge_detection(image):
        return (torch.gradient(image.squeeze(0).squeeze(0))[0]**2 + 
                torch.gradient(image.squeeze(0).squeeze(0))[1]**2
                ).unsqueeze(0).unsqueeze(0)