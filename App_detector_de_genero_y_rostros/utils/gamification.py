
# utils/gamification.py

import tkinter as tk
from tkinter import messagebox

class Gamification:
    def __init__(self, points_label, badges_label):
        self.points = 0
        self.badges = set()
        self.points_label = points_label
        self.badges_label = badges_label

    def add_points(self, amount, parent=None):
        self.points += amount
        self._check_badges(parent)
        self._update_ui()

    def _check_badges(self, parent):
        new_badge = None
        if self.points >= 50 and "Explorador" not in self.badges:
            self.badges.add("Explorador")
            new_badge = "Explorador"
        if self.points >= 100 and "Experto en filtros" not in self.badges:
            self.badges.add("Experto en filtros")
            new_badge = "Experto en filtros"
        if self.points >= 200 and "Maestro de mapas" not in self.badges:
            self.badges.add("Maestro de mapas")
            new_badge = "Maestro de mapas"
        
        if new_badge and parent:
            messagebox.showinfo("¡Logro desbloqueado!", f"¡Has obtenido el badge '{new_badge}'!")

    def _update_ui(self):
        self.points_label.config(text=f"🏆 {self.points} pts")
        badges_text = " ".join([f"🏅{b}" for b in self.badges])
        self.badges_label.config(text=badges_text)