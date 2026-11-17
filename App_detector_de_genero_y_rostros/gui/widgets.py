# gui/widgets.py

import tkinter as tk

def create_button(parent, text, color, command):
    btn = tk.Button(parent, text=text, bg=color, fg="white", padx=12, pady=5,
                    relief=tk.FLAT, font=('Segoe UI', 9, 'bold'),
                    activebackground=color, activeforeground="white",
                    command=command)
    return btn