import mss, pytesseract, os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# Region around the Discord button (adjust if needed)
left = 201
top = 896
width = 400
height = 120

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Ajing\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

sct = mss.mss()
rect = {"left": left, "top": top, "width": width, "height": height}
print("Region:", rect)

sct_img = sct.grab(rect)
img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
try:
    img.save('ocr_region_raw.png')
    print('Saved region image to ocr_region_raw.png')
except Exception as e:
    print('Could not save image:', e)


def proc(name, img, scale=1, threshold=None, contrast=1.0, sharpen=False, psm_list=(6,7)):
    im = img
    if scale != 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    img_gray = ImageOps.grayscale(im)
    if contrast != 1.0:
        img_gray = ImageEnhance.Contrast(img_gray).enhance(contrast)
    if sharpen:
        img_gray = img_gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    if threshold is not None:
        img_proc = img_gray.point(lambda p: 255 if p > threshold else 0)
    else:
        img_proc = img_gray

    fname = f"{name}_s{scale}_c{contrast}_thr{threshold}_sh{int(sharpen)}.png"
    try:
        img_proc.save(fname)
        print('Saved processed image:', fname)
    except Exception as e:
        print('Could not save processed image:', e)

    for psm in psm_list:
        config = f'--psm {psm}'
        try:
            data = pytesseract.image_to_data(img_proc, output_type=pytesseract.Output.DICT, config=config)
        except Exception as e:
            print('Tesseract error:', e)
            continue
        texts = data.get('text', [])
        confs = data.get('conf', [])
        lefts = data.get('left', [])
        tops = data.get('top', [])
        widths = data.get('width', [])
        heights = data.get('height', [])
        print(f"--- {name} scale={scale} contrast={contrast} thr={threshold} sharpen={sharpen} psm={psm} ---")
        for i, t in enumerate(texts):
            if not str(t).strip():
                continue
            conf = confs[i] if i < len(confs) else None
            l = lefts[i] if i < len(lefts) else None
            tp = tops[i] if i < len(tops) else None
            w = widths[i] if i < len(widths) else None
            h = heights[i] if i < len(heights) else None
            print(i, repr(t), conf, l, tp, w, h)


variations = [
    (1, None, 1.0, False),
    (2, None, 1.0, False),
    (3, None, 1.0, False),
    (2, 150, 1.0, False),
    (2, 150, 1.0, True),
    (2, None, 1.5, False),
    (2, None, 2.0, False),
]

for idx, (scale, thr, contrast, sharpen) in enumerate(variations):
    proc(f"v{idx}", img, scale=scale, threshold=thr, contrast=contrast, sharpen=sharpen)

print('Done.')
