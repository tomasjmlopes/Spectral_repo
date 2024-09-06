import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import numpy as np
from .affine_transform import AffineTransform

def visualize_affine_transform_with_sliders(mod1, mod2):
    """
    Visualizes the affine transformation of mod1 to align with mod2 using sliders to control
    scale, translation, and rotation.

    Parameters:
    mod1: numpy.ndarray
        The image to be transformed (first image).
    mod2: numpy.ndarray
        The target image (second image) to overlay the transformed mod1 on.

    Returns:
    dict: Final parameters of the affine transformation.
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.4)

    init_scale_x = 1.0
    init_scale_y = 1.0
    init_translation_x = 0.0
    init_translation_y = 0.0
    init_angle = 0

    affine_transform = AffineTransform(init_scale_x, init_scale_y, init_translation_x, init_translation_y, init_angle)
    to_transform = torch.tensor(mod1).unsqueeze(0).unsqueeze(0).float()
    target_image = torch.tensor(mod2).unsqueeze(0).unsqueeze(0).float()

    transformed_img1_tensor = affine_transform(to_transform, target_image.size())
    transformed_img1 = transformed_img1_tensor.squeeze(0).squeeze(0).detach().numpy()

    img_plot = ax.imshow(mod2, cmap='gray', alpha=0.5)
    img_plot_transformed = ax.imshow(transformed_img1, cmap='jet', alpha=0.5)

    ax_scale_x = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_scale_y = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_translation_x = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_translation_y = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_angle = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    slider_scale_x = Slider(ax_scale_x, 'Scale X', 0.2, 10, valinit=init_scale_x)
    slider_scale_y = Slider(ax_scale_y, 'Scale Y', 0.2, 10, valinit=init_scale_y)
    slider_translation_x = Slider(ax_translation_x, 'Trans X', -1, 1, valinit=init_translation_x)
    slider_translation_y = Slider(ax_translation_y, 'Trans Y', -1, 1, valinit=init_translation_y)
    slider_angle = Slider(ax_angle, 'Angle', -np.pi, np.pi, valinit=init_angle)

    final_params = {
        'scale_x': init_scale_x,
        'scale_y': init_scale_y,
        'translate': [init_translation_x, init_translation_y],
        'rot_angle': init_angle
    }

    def update(val):
        scale_x = slider_scale_x.val
        scale_y = slider_scale_y.val
        translation_x = slider_translation_x.val
        translation_y = slider_translation_y.val
        angle = slider_angle.val

        affine_transform = AffineTransform(scale_x, scale_y, translation_x, translation_y, angle)
        transformed_img1_tensor = affine_transform(to_transform, target_image.size())
        transformed_img1 = transformed_img1_tensor.squeeze(0).squeeze(0).detach().numpy()

        img_plot_transformed.set_data(transformed_img1)
        fig.canvas.draw_idle()

        final_params['scale_x'] = scale_x
        final_params['scale_y'] = scale_y
        final_params['translate'] = [translation_x, translation_y]
        final_params['rot_angle'] = angle

    slider_scale_x.on_changed(update)
    slider_scale_y.on_changed(update)
    slider_translation_x.on_changed(update)
    slider_translation_y.on_changed(update)
    slider_angle.on_changed(update)

    plt.show()

    return final_params