import numpy as np
from matplotlib.pyplot import *
import matplotlib.colors as colors

class align:
    def __init__(
            self, 
            image,
            image_to_transform,
            figsize = (12, 6),
            lognorm = True,
            vmax_r = 1,
            vmin_r = 0,
            vmax_l = 1,
            vmin_l = 0,
            pixel_tol = 2,
            sig = 0.1,
            weigths = False
    ):
        """
        Variable Initialization. Takes as input two 2D datasets (or RGB images)
        we wish to align

        Parameters:
            image: array_like (grayscale or RGB)
                Reference image
            image_to_transform: array_like (grayscale or RGB)
                Image we wish to transform
            figsize: list
                Size of the plotting area
        """
        self.S = None
        self.V = None
        self.U = None
        self.T = None
        self.weigths = weigths
        self.sig = sig
        self.pixel_tol = pixel_tol
        self.centroid = None
        self.point_cloud_image = []
        self.point_cloud_image_to_tranform = []
        self.new_data = None
        self.x_size, self.y_size = image.shape[0], image.shape[1]
        self.image_to_tranform = image_to_transform

        ### Plotting ###
        self.fig, self.ax = subplots(1, 3, figsize = figsize)

        if lognorm:
            self.ax[0].imshow(image, interpolation = 'None', norm = colors.LogNorm(vmin = 10, vmax = image.max()), origin = 'lower')
        else:
            self.ax[0].imshow(image, interpolation = 'None', origin = 'lower', vmin = vmin_l, vmax = vmax_l)

        self.plot1, = self.ax[0].plot([], [], color = 'r', marker = 'o', ms = 5)
        self.ax[0].set_title('Reference Image')

        self.ax[1].imshow(image_to_transform, interpolation = 'None', origin = 'lower', vmin = vmin_r, vmax = vmax_r)
        self.plot2, = self.ax[1].plot([], [], color = 'r', marker = 'o', ms = 5)
        self.ax[1].set_title('Image to Tansform')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _right_click(self, event):
        """
        Point Selection
        """
        if event.inaxes == self.ax[0]: #warrants that we are clicking the correct axes
            self.point_cloud_image.append([event.xdata, event.ydata])
            self.plot1.set_data([row[0] for row in self.point_cloud_image], [row[1] for row in self.point_cloud_image])
    
        if event.inaxes == self.ax[1]: #warrants that we are clicking the correct axes
            self.point_cloud_image_to_tranform.append([event.xdata, event.ydata])
            self.plot2.set_data([row[0] for row in self.point_cloud_image_to_tranform], [row[1] for row in self.point_cloud_image_to_tranform])
    
    def _left_click(self, event):
        """
        Alignment action
        """
        if len(self.point_cloud_image) != len(self.point_cloud_image_to_tranform):
            print("The number of points selected should be the same for both images")
        else:
            left_plot = np.array(self.point_cloud_image)
            right_plot = np.array(self.point_cloud_image_to_tranform)

            # Rough Alignment
            self.centroid, self.T, self.U, self.S, self.V = self._svd_params(left_plot, right_plot)
            #Transform the data
            self.transform(self.image_to_tranform)
            t_p = np.array([self.T + b @ self.U @ np.diag(self.S) @ self.V for b in (left_plot - self.centroid)])
            self.ax[2].imshow(self.new_data, interpolation = 'None', origin = 'lower')
            self.ax[1].plot(t_p[:, 0], t_p[:, 1], color = 'k', marker = 'o')

    def make_gauss_weights(self, arr_shape, sigma):
        """
        Gaussian Weights for interpolation
        """
        XX, YY = np.meshgrid(np.arange(arr_shape[0]), np.arange(arr_shape[1]))
        XX_center, YY_center = (arr_shape[0] - 1)/2, (arr_shape[1] - 1)/2
        gw = np.exp(- ( (XX - XX_center)**2 + (YY - YY_center)**2 ) / (2 * sigma**2)) 
        return gw

    def transform(self, data):
        """
        Uses the found parameters to transform an image or an entire
        data cube. The transformed dataset with be stored in the variable 
        new_data.
        
        Parameters:
            data: array_like

        Returns:
            -----
        """
        if len(data.shape) == 2:
            self.new_data = np.zeros((self.x_size, self.y_size))
        elif data.shape[-1] == 3:
            self.new_data = np.zeros((self.x_size, self.y_size, data.shape[2]), dtype = int)
        else:
            self.new_data = np.zeros((self.x_size, self.y_size, data.shape[2]))

        gauss_ws = self.make_gauss_weights((int(self.pixel_tol*2), int(self.pixel_tol*2)), sigma = self.sig)
        nearest_ws = np.ones((int(self.pixel_tol*2), int(self.pixel_tol*2)))
        for i in range(0, self.y_size):
            for j in range(0, self.x_size):
                vec = self.T + np.array([i, j] - self.centroid) @ self.U @ np.diag(self.S) @ self.V
                new_i, new_j = int(vec[0]), int(vec[1])
                
                if self.weigths == 'gaussian':
                    try:
                        if len(data.shape) == 2:
                            patch = data[new_j-self.pixel_tol : new_j+self.pixel_tol, new_i-self.pixel_tol : new_i+self.pixel_tol]
                            self.new_data[j, i] = np.average(patch, weights = gauss_ws, axis = (0, 1))
                        else:
                            patch = data[new_j-self.pixel_tol : new_j+self.pixel_tol, new_i-self.pixel_tol : new_i+self.pixel_tol, :]
                            self.new_data[j, i, :] = np.average(patch, weights = gauss_ws, axis = (0, 1))
                    except:
                        pass

                elif self.weigths == 'nearest':
                    try:
                        if len(data.shape) == 2:
                            patch = data[new_j-self.pixel_tol : new_j+self.pixel_tol, new_i-self.pixel_tol : new_i+self.pixel_tol]
                            self.new_data[j, i] = np.average(patch, axis = (0, 1), weights = nearest_ws)
                        else:
                            patch = data[new_j-self.pixel_tol : new_j+self.pixel_tol, new_i-self.pixel_tol : new_i+self.pixel_tol, :]
                            self.new_data[j, i, :] = np.average(patch, axis = (0, 1), weights = nearest_ws)
                    except:
                        pass

                else:
                    try:
                        if len(data.shape) == 2:
                            self.new_data[j, i] = data[new_j, new_i]
                        else:
                            self.new_data[j, i, :] = data[new_j, new_i]
                    except:
                        pass
            
    def _svd_params(self, A, B):
        """
        Align centroids and use SVD to find transformation
        parameters. Very similar to a regression.

        Parameters:
            A: array_like
                point cloud we wish to align
            B: array_like
                reference point cloud

        Returns:
            Rotation: array_like
                3D rotation matrix 
            Scale: float
                Scaling factor
            Translation: array_like
                Translation vector
        """
        assert A.shape == B.shape
        
        A_center = np.mean(A, axis = 0)
        B_center = np.mean(B, axis = 0)

        centered_A = A - A_center
        centered_B = B - B_center

        X = np.linalg.inv(centered_A.T @ centered_A) @ centered_A.T @ centered_B
        U, S, V = np.linalg.svd(X)
        return A_center, B_center, U, S, V
    
    def _on_click(self, event):
        """
        Button press actions
        """
        if event.button == MouseButton.RIGHT:
            self._right_click(event)

        elif  event.dblclick:
            self._left_click(event)