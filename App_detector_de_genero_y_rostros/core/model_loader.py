# core/model_loader.py

import tensorflow as tf
import threading
import zipfile
import os

class ModelLoader:
    def __init__(self):
        self.model = None
        self.base_model = None
        self.loaded = False
        self._lock = threading.Lock()

    def load(self, filepath, callback=None):
        def _load():
            h5_path = filepath + ".h5"
            try:
                # 1. Reconstruir la arquitectura EXACTA basada en tu lista
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(128, 128, 3),
                    include_top=False,
                    weights=None, # No descargar, usaremos los tuyos
                    alpha=1.0
                )
                base_model._name = 'mobilenetv2_1.00_128' # Forzar nombre original

                model = tf.keras.Sequential([
                    tf.keras.Input(shape=(128, 128, 3), name='input_layer_2'),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d'),
                    tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
                    tf.keras.layers.Dropout(0.5, name='dropout_1'),
                    tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')
                ])

                # 2. Extraer los pesos puros del archivo .keras (es un ZIP)
                if zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath, 'r') as z:
                        for name in z.namelist():
                            if 'weights' in name and name.endswith('.h5'):
                                with open(h5_path, 'wb') as f:
                                    f.write(z.read(name))
                                break
                else:
                    h5_path = filepath

                # 3. Inyectar los pesos ignorando errores menores de nombres
                model.load_weights(h5_path, by_name=True, skip_mismatch=True)

                # 4. Limpiar el archivo temporal
                if h5_path != filepath and os.path.exists(h5_path):
                    os.remove(h5_path)

                with self._lock:
                    self.model = model
                    self.base_model = base_model
                    self.loaded = True
                
                if callback:
                    callback(True, None)

            except Exception as e:
                if callback:
                    callback(False, f"Error al inyectar pesos: {str(e)}")

        threading.Thread(target=_load, daemon=True).start()

    def predict(self, image_batch):
        if not self.loaded or self.model is None:
            raise RuntimeError("Modelo no cargado")
        return self.model.predict(image_batch, verbose=0)

    def get_base_model(self):
        return self.base_model