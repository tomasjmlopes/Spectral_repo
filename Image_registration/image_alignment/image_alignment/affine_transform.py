import torch
from torch import nn
from torch.functional import F

class AffineTransform(nn.Module):
    def __init__(self,
                 scale_x_pred, scale_y_pred,
                 translation_x_pred, translation_y_pred,
                 angle_pred,
                 s_tol = 0, t_tol = 0, a_tol = 0,
                 device = 'cpu'):
        super().__init__()
        self.scalex = nn.Parameter(torch.tensor(scale_x_pred, dtype=torch.float, device=device))
        self.scaley = nn.Parameter(torch.tensor(scale_y_pred, dtype=torch.float, device=device))
        self.transX = nn.Parameter(torch.tensor(translation_x_pred, dtype=torch.float, device=device))
        self.transY = nn.Parameter(torch.tensor(translation_y_pred, dtype=torch.float, device=device))
        self.angle = nn.Parameter(torch.tensor(angle_pred, dtype=torch.float, device=device))
        self.device = device
        self.task = 'bilinear'

        if scale_x_pred - s_tol < 0.1:
            self.scale_x_bounds = (0.1, scale_x_pred + s_tol)
        else:
            self.scale_x_bounds = (scale_x_pred - s_tol, scale_x_pred + s_tol)

        if scale_y_pred - s_tol < 0.1:
            self.scale_y_bounds = (0.1, scale_y_pred + s_tol)
        else:
            self.scale_y_bounds = (scale_y_pred - s_tol, scale_y_pred + s_tol)

        self.translation_x_bounds = (translation_x_pred - t_tol, translation_x_pred + t_tol)
        self.translation_y_bounds = (translation_y_pred - t_tol, translation_y_pred + t_tol)
        self.angle_bounds = (angle_pred - a_tol, angle_pred + a_tol)

    def get_theta(self):
        cos_angle = torch.cos(self.angle)
        sin_angle = torch.sin(self.angle)

        # Create rotation matrix (3x3)
        R = torch.stack([
            torch.stack([cos_angle, -sin_angle, torch.zeros(1, device=self.device).squeeze()]),
            torch.stack([sin_angle, cos_angle, torch.zeros(1, device=self.device).squeeze()]),
            torch.tensor([0., 0., 1.], device=self.device)
        ])

        # Create scaling matrix (3x3)
        S = torch.stack([
            torch.stack([self.scalex, torch.zeros(1, device=self.device).squeeze(), torch.zeros(1, device=self.device).squeeze()]),
            torch.stack([torch.zeros(1, device=self.device).squeeze(), self.scaley, torch.zeros(1, device=self.device).squeeze()]),
            torch.tensor([0., 0., 1.], device=self.device)
        ])

        # Create translation matrix (3x3)
        T = torch.stack([
            torch.stack([torch.ones(1, device=self.device).squeeze(), torch.zeros(1, device=self.device).squeeze(), self.transX]),
            torch.stack([torch.zeros(1, device=self.device).squeeze(), torch.ones(1, device=self.device).squeeze(), self.transY]),
            torch.tensor([0., 0., 1.], device=self.device)
        ])

        M = torch.matmul(torch.matmul(T, R), S)
        #M_inv = torch.inverse(M)
        return M[:2, :].unsqueeze(0)

    def forward(self, x, y_size):
        self.scalex.data = torch.clamp(self.scalex.data, self.scale_x_bounds[0], self.scale_x_bounds[1])
        self.scaley.data = torch.clamp(self.scaley.data, self.scale_y_bounds[0], self.scale_y_bounds[1])
        self.transX.data = torch.clamp(self.transX.data, self.translation_x_bounds[0], self.translation_x_bounds[1])
        self.transY.data = torch.clamp(self.transY.data, self.translation_y_bounds[0], self.translation_y_bounds[1])
        self.angle.data = torch.clamp(self.angle.data, self.angle_bounds[0], self.angle_bounds[1])

        theta = self.get_theta()
        grid = F.affine_grid(theta, y_size, align_corners = False)
        return F.grid_sample(x, grid, align_corners = False, mode = self.task)