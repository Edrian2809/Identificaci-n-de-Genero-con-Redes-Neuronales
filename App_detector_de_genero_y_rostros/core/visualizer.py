# core/visualizer.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualizer:
    @staticmethod
    def get_filters(base_model, max_filters=16):
        """Extrae los filtros de la primera capa convolucional."""
        first_conv = None
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                first_conv = layer
                break
        if first_conv is None:
            raise ValueError("No se encontró capa Conv2D en el modelo base.")

        weights = first_conv.get_weights()
        if len(weights) < 1:
            raise ValueError("La capa no tiene pesos.")
        filters = weights[0]  # (H, W, in_channels, out_channels)
        f_min, f_max = filters.min(), filters.max()
        filters_norm = (filters - f_min) / (f_max - f_min + 1e-8)
        return filters_norm[:, :, :, :max_filters]

    @staticmethod
    def get_feature_maps(base_model, image_batch):
        """Obtiene los mapas de características de la primera capa convolucional."""
        first_conv = None
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                first_conv = layer
                break
        if first_conv is None:
            raise ValueError("No se encontró capa Conv2D.")
        intermediate = tf.keras.Model(inputs=base_model.input, outputs=first_conv.output)
        return intermediate.predict(image_batch, verbose=0)[0]  # (H, W, C)

    @staticmethod
    def plot_filters(filters, master, num_cols=4, cmap='gray'):
        num_filters = filters.shape[-1]
        num_cols = min(num_cols, num_filters)
        num_rows = (num_filters + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.patch.set_facecolor('#0B0B1A')

        for i in range(num_filters):
            r = i // num_cols
            c = i % num_cols
            filtro = filters[:, :, :, i]
            filtro_gray = np.mean(filtro, axis=2)  # promediar canales de entrada
            axes[r, c].imshow(filtro_gray, cmap=cmap)
            axes[r, c].set_title(f'Filtro {i+1}', fontsize=9, color='white')
            axes[r, c].axis('off')
        # Ocultar ejes vacíos
        for i in range(num_filters, num_rows * num_cols):
            r = i // num_cols
            c = i % num_cols
            axes[r, c].axis('off')

        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        return canvas

    @staticmethod
    def plot_feature_maps(feature_maps, master, num_cols=4, cmap='viridis'):
        num_maps = feature_maps.shape[-1]
        num_cols = min(num_cols, num_maps)
        num_rows = (num_maps + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.patch.set_facecolor('#0B0B1A')

        for i in range(num_maps):
            r = i // num_cols
            c = i % num_cols
            fm = feature_maps[:, :, i]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            axes[r, c].imshow(fm_norm, cmap=cmap)
            axes[r, c].set_title(f'Mapa {i+1}', fontsize=9, color='white')
            axes[r, c].axis('off')
        for i in range(num_maps, num_rows * num_cols):
            r = i // num_cols
            c = i % num_cols
            axes[r, c].axis('off')

        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        return canvas