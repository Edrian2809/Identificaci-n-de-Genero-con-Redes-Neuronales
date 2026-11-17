# core/face_detector.py

import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        # Usar tu archivo .onnx de YuNet
        model_path = "face_detection_yunet_2023mar.onnx"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo {model_path} en la raíz del proyecto.")
            
        # Cargar el detector YuNet usando OpenCV
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

    def detect(self, image_bgr):
        """Detecta rostros humanos en imagen BGR. Retorna (is_human, image_with_boxes)."""
        h, w, _ = image_bgr.shape
        
        # YuNet necesita saber el tamaño de la imagen para procesarla bien
        self.detector.setInputSize((w, h))
        
        # Realizar detección
        _, faces = self.detector.detect(image_bgr)
        
        is_human = faces is not None and len(faces) > 0
        img_copy = image_bgr.copy()

        if is_human:
            for face in faces:
                # face contiene [x, y, w, h, ... puntos de referencia]
                box = face[0:4].astype(int)
                x, y, box_w, box_h = box
                
                # Dibujar rectángulo verde
                cv2.rectangle(img_copy, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

        return is_human, img_copy