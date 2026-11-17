
# core/predictor.py

import time
import numpy as np

class GenderPredictor:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def predict(self, image_batch):
        """Retorna (probabilidad_de_hombre, tiempo_ms)"""
        start = time.time()
        prob = self.model_loader.predict(image_batch)[0][0]
        elapsed_ms = (time.time() - start) * 1000
        return prob, elapsed_ms

    def classify(self, prob, threshold=0.5):
        if prob > threshold:
            return "Hombre", prob
        else:
            return "Mujer", 1 - prob