# gui/styles.py

from utils.constants import COLORS
import tkinter as tk
from tkinter import ttk

def apply_style(root):
    # Configuración básica de la ventana
    root.configure(bg=COLORS["bg"])
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background=COLORS["bg"], borderwidth=0)
    style.configure('TNotebook.Tab', background=COLORS["card"], foreground=COLORS["text_secondary"],
                    padding=[15, 8], font=('Segoe UI', 10, 'bold'))
    style.map('TNotebook.Tab', background=[('selected', COLORS["primary"])],
              foreground=[('selected', COLORS["bg"])])
    style.configure('TFrame', background=COLORS["bg"])
    style.configure('TLabel', background=COLORS["bg"], foreground=COLORS["text"])
    style.configure('TLabelframe', background=COLORS["card"], foreground=COLORS["text"],
                    borderwidth=1, relief='solid', bordercolor=COLORS["card_border"])
    style.configure('TLabelframe.Label', background=COLORS["card"], foreground=COLORS["primary"],
                    font=('Segoe UI', 10, 'bold'))
    # Barras de progreso con colores
    style.configure('green.Horizontal.TProgressbar', troughcolor=COLORS["card"], background=COLORS["accent"])
    style.configure('orange.Horizontal.TProgressbar', troughcolor=COLORS["card"], background=COLORS["warning"])
    style.configure('red.Horizontal.TProgressbar', troughcolor=COLORS["card"], background=COLORS["danger"])