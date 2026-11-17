# main.py  —  DEBE ir como PRIMERA línea
import matplotlib
matplotlib.use("Agg")            # backend no interactivo, seguro para Tk
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # <-- ESTO ARREGLA EL ERROR DEL MODELO
# ... (el resto de tu main.py sigue igual)
import tkinter as tk
from gui.main_window import MainWindow

if __name__ == "__main__":
    print("🚀 Iniciando AI Vision Lab...", flush=True)
    root = tk.Tk()
    app = MainWindow(root)
    print("✅ Aplicación iniciada. Ejecutando mainloop...", flush=True)
    root.mainloop()
    print("👋 Aplicación cerrada.", flush=True)