import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

class NoScale:
    def __init__(self):
        pass

    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x

class NonlinearRegressionAnalysis:
    def __init__(self, X, y, patch_shape, test_size=0.2, random_state=42, scaler=None):
        self.X = X
        self.y = y

        self.patch_shape = patch_shape
        if scaler == None:
            self.scaler = NoScale()
        elif scaler == 'MinMax':
            self.scaler = MinMaxScaler()
        elif scaler == 'MaxAbs':
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Scaler type not available. Choose between MinMax, MaxAbs, or None for no scaler")
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        self.mape = None
        self.mae = None
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.reconstruction = None
        self.custom_test_data = None
        self.custom_test_pred = None
        self.custom_patch_shape = None

    def calculate_metrics(self, y_true, y_pred):
        percentage_error = ((y_true - y_pred) / y_true) * 100
        mape = np.abs(percentage_error).mean()
        mae = np.abs(y_true - y_pred).mean()
        return percentage_error, mape, mae

    def train_model(self, model_type='knn', **kwargs):
        if model_type == 'knn':
            self.model = KNeighborsRegressor(**kwargs)
            self.model_name = 'K-Nearest Neighbors'
        elif model_type == 'kernel_ridge':
            self.model = KernelRidge(**kwargs)
            self.model_name = 'Kernel Ridge'
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**kwargs)
            self.model_name = 'Random Forest'
        elif model_type == 'svr':
            self.model = SVR(**kwargs)
            self.model_name = 'Support Vector Regression'
        elif model_type == 'gbr':
            self.model = GradientBoostingRegressor(**kwargs)
            self.model_name = 'Gradient Boosting'
        elif model_type == 'mlp':
            self.model = MLPRegressor(**kwargs)
            self.model_name = 'Multi-Layer Perceptron'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.fit(self.X_train, self.y_train)
        self._predict()

    def _predict(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        self.reconstruction = self.model.predict(self.scaler.transform(self.X)).reshape(self.patch_shape)

    def apply_to_new_data(self, X_new, patch_shape=None):
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        if patch_shape is None:
            patch_shape = self.patch_shape
        
        X_new_scaled = self.scaler.transform(X_new)
        y_new_pred = self.model.predict(X_new_scaled)
        reconstruction = y_new_pred.reshape(patch_shape)
        
        return reconstruction

    def set_custom_test_data(self, X_test, y_test, patch_shape):
        self.custom_test_data = (X_test, y_test)
        self.custom_patch_shape = patch_shape
        X_test_scaled = self.scaler.transform(X_test)
        self.custom_test_pred = self.model.predict(X_test_scaled)

    def plot_fill_between(self, ax, y_pred, mean_values, colors):
        x_vals = np.linspace(min(y_pred), max(y_pred), 100)
        for mean, color, alpha in zip(mean_values, colors, [0.3, 0.2, 0.2]):
            ax.fill_between(x_vals, 
                            x_vals * (1 - mean / 100), 
                            x_vals * (1 + mean / 100), 
                            color=color, alpha=alpha, label=f'{mean}% Mean Percentage Error')

    def plot_histogram(self, ax, data='original'):
        if data == 'original':
            train_percentage_error, _, _ = self.calculate_metrics(self.y_train, self.y_train_pred)
            test_percentage_error, self.mape, self.mae = self.calculate_metrics(self.y_test, self.y_test_pred)
        elif data == 'custom' and self.custom_test_data is not None:
            _, y_test = self.custom_test_data
            train_percentage_error, _, _ = self.calculate_metrics(self.y_train, self.y_train_pred)
            test_percentage_error, self.mape, self.mae = self.calculate_metrics(y_test, self.custom_test_pred)
        else:
            raise ValueError("Invalid data type or custom test data not set")

        ax.hist(test_percentage_error, bins=100, alpha=0.7, label="Test", range=(-400, 400), density=True)
        ax.hist(train_percentage_error, bins=100, alpha=0.7, label="Train", range=(-400, 400), density=True)

        ax.set_title("Histogram of Percentage Errors", fontsize=14)
        ax.set_xlabel("Error (%)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        textstr = f"MAPE: {self.mape:.2f}%\nMAE: {self.mae:.2f} ppm"
        ax.text(1, 0.4, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

    def plot_reference(self, ax, data='original'):
        if data == 'original':
            y_reshaped = self.y.reshape(self.patch_shape)
            patch_shape = self.patch_shape
        elif data == 'custom' and self.custom_test_data is not None:
            _, y_test = self.custom_test_data
            y_reshaped = y_test.reshape(self.custom_patch_shape)
            patch_shape = self.custom_patch_shape
        else:
            raise ValueError("Invalid data type or custom test data not set")

        vmax = y_reshaped.max()
        vmin = y_reshaped.min()
        im = ax.imshow(y_reshaped.T, cmap='turbo', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
        ax.set_title(f"ICP-MS", fontsize=16)
        ax.set_ylabel("Pixels", fontsize=14)

        ax.set_xticks([0, patch_shape[0]])
        ax.set_yticks([0, patch_shape[1]])
        ax.tick_params(axis='both', labelsize=12)
        
        return im

    def plot_reconstruction(self, ax, data='original'):
        if data == 'original':
            reconstruction = self.reconstruction
            patch_shape = self.patch_shape
            y_reshaped = self.y.reshape(self.patch_shape)
        elif data == 'custom' and self.custom_test_data is not None:
            reconstruction = self.custom_test_pred.reshape(self.custom_patch_shape)
            patch_shape = self.custom_patch_shape
            _, y_test = self.custom_test_data
            y_reshaped = y_test.reshape(self.custom_patch_shape)
        else:
            raise ValueError("Invalid data type or custom test data not set")

        vmax = y_reshaped.max()
        vmin = y_reshaped.min()
        im = ax.imshow(reconstruction.T, cmap='turbo', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
        ax.set_title(f"LIBS Prediction", fontsize=16)
        ax.set_xlabel("Pixels", fontsize=14)

        ax.set_xticks([0, patch_shape[0]])
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=12)

        return im

    def plot_error_map(self, ax, data='original'):
        if data == 'original':
            reconstruction = self.reconstruction
            patch_shape = self.patch_shape
            y_reshaped = self.y.reshape(self.patch_shape)
        elif data == 'custom' and self.custom_test_data is not None:
            reconstruction = self.custom_test_pred.reshape(self.custom_patch_shape)
            patch_shape = self.custom_patch_shape
            _, y_test = self.custom_test_data
            y_reshaped = y_test.reshape(self.custom_patch_shape)
        else:
            raise ValueError("Invalid data type or custom test data not set")

        abs_error = reconstruction - y_reshaped
        im = ax.imshow(abs_error.T, cmap='Reds', aspect='equal', origin='lower')
        ax.set_title(f"Error", fontsize=16)

        ax.set_xticks([0, patch_shape[0]])
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=12)

        return im

    def plot_actual_vs_predicted(self, ax, data='train'):
        mean_values = [25, 50, 75]
        colors = ['green', 'yellow', 'orange']
        
        if data == 'train':
            y_pred, y_true = self.y_train_pred, self.y_train
            color, label = 'red', 'Train data'
        elif data == 'test':
            y_pred, y_true = self.y_test_pred, self.y_test
            color, label = 'blue', 'Test data'
        elif data == 'custom' and self.custom_test_data is not None:
            _, y_true = self.custom_test_data
            y_pred = self.custom_test_pred
            color, label = 'purple', 'Custom test data'
        else:
            raise ValueError("Invalid data type or custom test data not set")
        
        self.plot_fill_between(ax, y_pred, mean_values, colors)
        ax.plot(y_pred, y_true, color=color, label=label, marker='o', ms=0.02, ls='')
        
        y_min, y_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([y_min, y_max], [y_min, y_max], color='k', linestyle='--', label='Perfect Prediction Line')
        ax.set_title(data.capitalize(), fontsize=12)
        ax.set_xlabel("LIBS Predicted Values", fontsize=10)
        if data == 'train':
            ax.set_ylabel("ICP-MS", fontsize=10)
        ax.set_xlim(0, y_max * 0.6)
        ax.set_ylim(0, y_max * 0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            axis.get_major_formatter().set_scientific(True)
            axis.get_major_formatter().set_powerlimits((0, 1))
        
        ax.figure.canvas.draw()
        offsetx = ax.xaxis.get_offset_text()
        offsetx.set_size(8)
        offsetx.set_position((1.2, -1))
        offsetx.set_va('bottom')

        offsety = ax.yaxis.get_offset_text()
        offsety.set_size(8)
        offsety.set_position((-0.3, 0.7))
        offsety.set_va('bottom')

    def add_colorbar(self, fig, ax, im, scientific = True):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.offsetText.set(size=10) 
        cbar.ax.yaxis.offsetText.set_position((6, -6)) 
        if scientific:
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
        cbar_min, cbar_max = im.get_clim()
        cbar.update_ticks()
        return cbar