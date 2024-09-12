import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

class no_scale:
    def __init__(self):
        pass

    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x

class NonlinearRegressionAnalysis:
    def __init__(self, X, y, patch_shape, test_size=0.2, random_state=42, scaler = None):
        self.X = X
        self.y = y
        self.patch_shape = patch_shape
        if scaler == None:
            self.scaler = no_scale()
        elif scaler == 'MinMax':
            self.scaler = MinMaxScaler()
        elif scaler == 'MaxAbs':
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Scaler type not available. Choose between MinMax, MaxAbs, or None for no scaler")
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, shuffle = True
        )
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.reconstruction = None
        self.custom_test_data = None
        self.custom_test_pred = None
        self.custom_patch_shape = None

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

    def get_metrics(self, data='original'):
        if data == 'original':
            train_mse = mean_squared_error(self.y_train, self.y_train_pred)
            test_mse = mean_squared_error(self.y_test, self.y_test_pred)
            train_r2 = r2_score(self.y_train, self.y_train_pred)
            test_r2 = r2_score(self.y_test, self.y_test_pred)
        elif data == 'custom' and self.custom_test_data is not None:
            _, y_test = self.custom_test_data
            train_mse = train_r2 = None  # No train metrics for custom data
            test_mse = mean_squared_error(y_test, self.custom_test_pred)
            test_r2 = r2_score(y_test, self.custom_test_pred)
        else:
            raise ValueError("Invalid data type or custom test data not set")

        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

    def plot_fill_between(self, ax, y_pred, mean_values, colors):
        x_vals = np.linspace(min(y_pred), max(y_pred), 100)
        for mean, color, alpha in zip(mean_values, colors, [0.3, 0.2, 0.2]):
            ax.fill_between(x_vals, 
                            x_vals * (1 - mean / 100), 
                            x_vals * (1 + mean / 100), 
                            color=color, alpha=alpha, label=f'{mean}% Mean Percentage Error')

    def plot_histogram(self, ax=None, data='original'):
        if ax is None:
            fig, ax = plt.subplots()
        
        if data == 'original':
            train_percentage_error = ((self.y_train - self.y_train_pred) / self.y_train)*100
            test_percentage_error = ((self.y_test - self.y_test_pred) / self.y_test)*100

            test_mape = (abs(self.y_test - self.y_test_pred) / self.y_test).mean() * 100
            test_mae = np.abs(self.y_test - self.y_test_pred).mean()

            ax.hist(test_percentage_error, bins=100, alpha=0.7, label="Test", range=(-400, 400), density=True)
            ax.hist(train_percentage_error, bins=100, alpha=0.7, label="Train", range=(-400, 400), density=True)
        elif data == 'custom' and self.custom_test_data is not None:
            _, y_test = self.custom_test_data
            test_percentage_error = ((y_test - self.custom_test_pred) / y_test)*100

            test_mape = (abs(y_test - self.custom_test_pred) / y_test).mean() * 100
            test_mae = np.abs(self.y_test - self.y_test_pred).mean()

            ax.hist(test_percentage_error, bins=100, alpha=0.7, label="Test", range=(-400, 400), density=True)
        else:
            raise ValueError("Invalid data type or custom test data not set")

        ax.set_title("Histogram of Percentage Errors", fontsize=14)
        ax.set_xlabel("Error (%)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adding a textbox
        textstr = "MAPE: {:.2f}%\nMAE: {:.2f}".format(test_mape, test_mae)
        ax.text(1, 0.4, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

        return ax

    def plot_reference(self, ax=None, data='original'):
        if ax is None:
            fig, ax = plt.subplots()

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

        # Set more spaced ticks on both axes
        ax.set_xticks([0, patch_shape[0]])
        ax.set_yticks([0, patch_shape[1]])
        ax.tick_params(axis='both', labelsize=12)
        
        return ax, im

    def plot_reconstruction(self, ax=None, data='original'):
        if ax is None:
            fig, ax = plt.subplots()

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

        # Set more spaced ticks on both axes
        ax.set_xticks([0, patch_shape[0]])
        ax.set_yticks([ ])
        ax.tick_params(axis='both', labelsize=12)

        return ax, im

    def plot_error_map(self, ax=None, data='original'):
        if ax is None:
            fig, ax = plt.subplots()

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

        abs_error = (reconstruction - y_reshaped)
        im = ax.imshow(abs_error.T, cmap='Reds', aspect='equal', origin='lower')
        ax.set_title(f"Error", fontsize=16)

        # Set more spaced ticks on both axes
        ax.set_xticks([0, patch_shape[0]])
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=12)

        return ax, im

    def plot_actual_vs_predicted(self, data='train', ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
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
        
        return ax

    def plot_all(self, data='original'):
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        if data == 'custom' and self.custom_test_data is None:
            raise ValueError("Custom test data not set. Call set_custom_test_data() first.")

        fig = plt.figure(figsize=(14, 4))
        gs = fig.add_gridspec(2, 5, width_ratios=[1.8, 1.8, 2, 2, 2], height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0:2])
        self.plot_histogram(ax1, data=data)

        ax2 = fig.add_subplot(gs[:, 2])
        ax2, im1 = self.plot_reference(ax2, data=data)

        ax3 = fig.add_subplot(gs[:, 3])
        ax3, im2 = self.plot_reconstruction(ax3, data=data)

        ax4 = fig.add_subplot(gs[:, 4])
        ax4, im3 = self.plot_error_map(ax4, data=data)

        if data == 'original':
            ax5 = fig.add_subplot(gs[1, 0])
            self.plot_actual_vs_predicted('train', ax5)

            ax6 = fig.add_subplot(gs[1, 1])
            self.plot_actual_vs_predicted('test', ax6)
        else:
            ax5 = fig.add_subplot(gs[1, 0:2])
            self.plot_actual_vs_predicted('custom', ax5)

        for ax, im in zip([ax2, ax3, ax4], [im1, im2, im3]):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            cbar = plt.colorbar(im, cax=cax)
            if ax == 'None':
                continue
            else:
                cbar.ax.yaxis.offsetText.set(size = 10) 
                cbar.ax.yaxis.offsetText.set_position((6, -5)) 
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.update_ticks()

        gs.tight_layout(fig, rect=[0, 0.0, 0.9, 1], h_pad=0.7, w_pad=0.1)  # h_pad affects vertical padding
        fig.align_ylabels([ax1, ax5])
        fig.align_ylabels([ax2, ax3, ax4])
        # fig.text(0.5, 1, f"Model: {self.model_name}", fontsize=20, ha='center', va='top')
        # fig.subplots_adjust(top=1)
        fig.suptitle(f"Model: {self.model_name}", fontsize=20, y=1.07)
        # fig.subplots_adjust(wspace=0.2, hspace=0.9)
        # fig.tight_layout()

        return fig
