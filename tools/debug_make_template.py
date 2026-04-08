import cv2
import numpy as np
import os

src = 'ocr_region_raw.png'
out = 'template_fish.png'
if not os.path.exists(src):
    print('Source image not found:', src)
    raise SystemExit(1)
img = cv2.imread(src)
if img is None:
    print('Failed to read', src)
    raise SystemExit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Purple-ish hue range (may need tuning)
lower = np.array([120, 40, 40])
upper = np.array([170, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    tpl = img[y0:y1, x0:x1]
    cv2.imwrite(out, tpl)
    print('Saved template to', out, 'rect=', (x0, y0, x1-x0, y1-y0))
else:
    # fallback: center crop
    h, w = img.shape[:2]
    cw, ch = 200, 60
    x0 = max(0, w//2 - cw//2)
    y0 = max(0, h//2 - ch//2)
    tpl = img[y0:y0+ch, x0:x0+cw]
    cv2.imwrite(out, tpl)
    print('Saved fallback template to', out, 'rect=', (x0, y0, cw, ch))
