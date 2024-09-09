import torch
from .affine_transform import AffineTransform
from .mi import MI
import numpy as np
from typing import Callable, Optional
from collections import defaultdict

class Lookahead:
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
            self.state["slow_weights"] = []
            for p in group['params']:
                self.state["slow_weights"].append(p.clone().detach())

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.optimizer.step()
        
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] % self.k == 0:
                for p, slow_p in zip(group["params"], self.state["slow_weights"]):
                    if p.grad is None:
                        continue
                    slow_p.add_(self.alpha * (p.data - slow_p))
                    p.data.copy_(slow_p)
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            "slow_weights": self.state["slow_weights"],
            "counters": [group["counter"] for group in self.param_groups]
        }
        return {
            "fast_state": fast_state_dict,
            "slow_state": slow_state
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = state_dict["fast_state"]
        slow_state = state_dict["slow_state"]
        self.optimizer.load_state_dict(fast_state_dict)
        self.state["slow_weights"] = slow_state["slow_weights"]
        for i, group in enumerate(self.param_groups):
            group["counter"] = slow_state["counters"][i]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return [group['lr'] for group in self.param_groups]

class AffineOptimizer:
    def __init__(self, image1, image2, initial_params, device='cpu', 
                 optim='SGD', lr=1e-4, momentum=0.85, num_iters=400, lookahead_k=5, lookahead_alpha=0.5):
        self.image1 = torch.tensor(image1).unsqueeze(0).unsqueeze(0).float().to(device)
        self.image2 = torch.tensor(image2).unsqueeze(0).unsqueeze(0).float().to(device)
        self.initial_params = initial_params
        self.device = device
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.num_iters = num_iters
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
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
            base_optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, maximize=True)
        elif self.optim == 'SGD':
            base_optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, maximize=True)
        else:
            raise ValueError('Optimizer not supported. Choose between Adam and SGD')

        # Wrap the optimizer with Lookahead
        optimizer = Lookahead(base_optimizer, k=self.lookahead_k, alpha=self.lookahead_alpha)

        scheduler = torch.optim.lr_scheduler.LinearLR(base_optimizer, start_factor=0.85, end_factor=1, total_iters=self.num_iters)

        best_params = None
        best_loss = float('-inf')

        for epoch in range(self.num_iters):
            optimizer.zero_grad()
            transformed_image = model(self.image1, self.image2.size())
            loss = self.mutual_info.mi(transformed_image, self.image2, mask=transformed_image != 0)

            print(f'Epoch {epoch}/{self.num_iters}, Current Loss: {loss:.6f}, Best Loss: {best_loss:.3f}', end='\r')

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

# class AffineOptimizer:
#     def __init__(self, image1, image2, initial_params, device='cpu', 
#                  optim='SGD', lr=1e-4, momentum=0.85, num_iters=400):
#         self.image1 = torch.tensor(image1).unsqueeze(0).unsqueeze(0).float().to(device)
#         self.image2 = torch.tensor(image2).unsqueeze(0).unsqueeze(0).float().to(device)
#         self.initial_params = initial_params
#         self.device = device
#         self.optim = optim
#         self.lr = lr
#         self.momentum = momentum
#         self.num_iters = num_iters
#         self.mutual_info = MI(dimension=2, num_bins=32, kernel_sigma=1).to(device)

#     def optimize(self, s_tol=0.2, t_tol=0.3, a_tol=np.pi/6, edge_align=False):
#         if edge_align:
#             self.image1 = self._edge_detection(self.image1)
#             self.image2 = self._edge_detection(self.image2)

#         model = AffineTransform(
#             scale_x_pred=self.initial_params['scale_x'],
#             scale_y_pred=self.initial_params['scale_y'],
#             translation_x_pred=self.initial_params['translate'][0],
#             translation_y_pred=self.initial_params['translate'][1],
#             angle_pred=self.initial_params['rot_angle'],
#             s_tol=s_tol, t_tol=t_tol, a_tol=a_tol,
#             device=self.device
#         )

#         if self.optim == 'Adam':
#             optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, maximize = True)
#         elif self.optim == 'SGD':
#             optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, maximize=True)
#         else:
#             raise ValueError('Optimizer not supported. Choose between Adam and SGD')

#         # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.85, end_factor=1, total_iters=self.num_iters)

#         best_params = None
#         best_loss = float('-inf')

#         for epoch in range(self.num_iters):
#             optimizer.zero_grad()
#             transformed_image = model(self.image1, self.image2.size())
#             loss = self.mutual_info.mi(transformed_image, self.image2, mask=transformed_image != 0)

#             print(f'Epoch {epoch}/{self.num_iters}, Current Loss: {loss:.6f}, Best Loss: {best_loss:.3f}', end = '\r')

#             if loss.item() > best_loss:
#                 best_loss = loss.item()
#                 best_params = {
#                     'rot_angle': model.angle.item(),
#                     'translate': [model.transX.item(), model.transY.item()],
#                     'scale_x': model.scalex.item(),
#                     'scale_y': model.scaley.item(),
#                 }
#             loss.backward()
#             optimizer.step()
#             # scheduler.step()

#         return best_params

#     @staticmethod
#     def _edge_detection(image):
#         return (torch.gradient(image.squeeze(0).squeeze(0))[0]**2 + 
#                 torch.gradient(image.squeeze(0).squeeze(0))[1]**2
#                 ).unsqueeze(0).unsqueeze(0)