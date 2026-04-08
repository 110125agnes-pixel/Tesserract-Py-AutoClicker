#!/usr/bin/env python3
import importlib

mods = ['tkinter', 'pytesseract', 'mss', 'pyautogui', 'keyboard', 'PIL', 'cv2', 'numpy']

for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m} OK")
    except Exception as e:
        print(f"{m} ERROR: {e!r}")
