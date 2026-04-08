import time
import ctypes
from ctypes import wintypes
import pyautogui

print('Move cursor to the Discord button now — waiting 5 seconds...')
time.sleep(5)
x,y = pyautogui.position()
pt = wintypes.POINT(x,y)
hWnd = ctypes.windll.user32.WindowFromPoint(pt)
length = ctypes.windll.user32.GetWindowTextLengthW(hWnd)
buf = ctypes.create_unicode_buffer(length+1)
ctypes.windll.user32.GetWindowTextW(hWnd, buf, length+1)
pid = wintypes.DWORD()
ctypes.windll.user32.GetWindowThreadProcessId(hWnd, ctypes.byref(pid))
print('POS:', x, y)
print('Window handle:', hWnd)
print('Window title:', buf.value)
print('ProcessId:', pid.value)
