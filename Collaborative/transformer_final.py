import numpy as np
from matplotlib.pyplot import *

class align:
    def __init__(
            self, 
            image,
            image_to_transform,
            figsize = (12, 6),
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
        self.centroid = None
        self.point_cloud_image = []
        self.point_cloud_image_to_tranform = []
        self.new_data = None
        self.x_size, self.y_size = image.shape[0], image.shape[1]
        self.image_to_tranform = image_to_transform

        ### Plotting ###
        self.fig, self.ax = subplots(1, 3, figsize = figsize)

        self.ax[0].imshow(image, interpolation = 'gaussian', origin = 'lower')
        self.plot1, = self.ax[0].plot([], [], color = 'r', marker = 'o', ms = 5)
        self.ax[0].set_title('Reference Image')

        self.ax[1].imshow(image_to_transform, interpolation = 'gaussian', origin = 'lower')
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
            self.ax[2].imshow(self.new_data, interpolation = 'gaussian', origin = 'lower')
            self.ax[1].plot(t_p[:, 0], t_p[:, 1], color = 'k', marker = 'o')



    def transform(self, data, val = 0):
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
            self.new_data = np.zeros((self.x_size, self.y_size)) + val
        elif data.shape[-1] == 3:
            self.new_data = np.zeros((self.x_size, self.y_size, data.shape[2]), dtype = int) + val
        else:
            self.new_data = np.zeros((self.x_size, self.y_size, data.shape[2])) + val
        for i in range(0, self.y_size):
            for j in range(0, self.x_size):
                vec = self.T + np.array([i, j] - self.centroid) @ self.U @ np.diag(self.S) @ self.V
                new_i, new_j = int(vec[0]), int(vec[1])
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

        elif  event.button == MouseButton.LEFT:
            self._left_click(event)