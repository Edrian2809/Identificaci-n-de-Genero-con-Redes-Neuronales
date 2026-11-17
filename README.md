# 🧠 AI Vision Lab - Clasificador de Género y Detección Facial

**AI Vision Lab** es una aplicación de escritorio interactiva y educativa desarrollada en Python. Su objetivo principal es demostrar conceptos avanzados de Visión por Computadora y Deep Learning de una forma visual y accesible.

La aplicación combina la detección de rostros de alta precisión con modelos de clasificación de género, permitiendo al usuario explorar "bajo el capó" de las redes neuronales convolucionales (CNN).

## ✨ Características Principales

*   **Detección de Rostros (YuNet):** Utiliza el modelo ONNX `face_detection_yunet_2023mar` mediante OpenCV para una detección facial rápida y precisa.
*   **Clasificación de Género (DeepFace):** Integración con DeepFace para obtener predicciones de género robustas y de alta precisión en imágenes reales.
*   **Visualización de Redes Neuronales:** 
    *   Inspección de los **kernels (filtros)** de la primera capa convolucional.
    *   Generación de **mapas de características** (activaciones) para entender qué regiones de la imagen influyen en la decisión.
*   **Análisis Matemático:** Pestaña dedicada al cálculo manual de las capas densas (GAP, Dense, Sigmoid) para fines educativos.
*   **Procesamiento por Lotes (Batch):** Capacidad de analizar múltiples imágenes a la vez.
*   **Gamificación:** Sistema de puntos y medallas para incentivar la exploración de las herramientas de la app.

## 🛠️ Tecnologías Utilizadas

*   **Lenguaje:** Python 3.12+
*   **Interfaz Gráfica:** Tkinter (CustomTkinter styling)
*   **Visión Artificial:** OpenCV (cv2), Pillow (PIL)
*   **Deep Learning:** TensorFlow 2.x, Keras, tf-keras (compatibilidad)
*   **Modelos Preentrenados:** DeepFace (Serengil), MobileNetV2, YuNet
*   **Visualización de Datos:** Matplotlib

## 🚀 Instalación y Ejecución

1. Clona este repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/AI-Vision-Lab.git
   cd AI-Vision-Lab
