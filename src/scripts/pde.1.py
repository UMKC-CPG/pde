#!/usr/bin/env python3

import vedo
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QColorDialog, QPushButton

app = QApplication([])

window = vedo.Window()

def change_color():
    color = QColorDialog().getColor()
    window.bgcolor(color.getRgbF())

button = QPushButton("Change Color")
button.clicked.connect(change_color)

layout = QVBoxLayout()
layout.addWidget(button)

widget = QWidget()
widget.setLayout(layout)
widget.show()

app.exec()
