import mss, pytesseract
from PIL import Image, ImageOps

# Region around the captured cursor position (Chrome window)
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
# save debug image to file for inspection
try:
    img.save('ocr_region.png')
    print('Saved region image to ocr_region.png')
except Exception as e:
    print('Could not save image:', e)

gray = ImageOps.grayscale(img)

def dump(name, im):
    print("---", name, "---")
    d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config='--psm 6')
    texts = d.get('text', [])
    confs = d.get('conf', [])
    lefts = d.get('left', [])
    tops = d.get('top', [])
    widths = d.get('width', [])
    heights = d.get('height', [])
    for i, txt in enumerate(texts):
        print(i, repr(txt), confs[i] if i < len(confs) else None, lefts[i] if i < len(lefts) else None, tops[i] if i < len(tops) else None, widths[i] if i < len(widths) else None, heights[i] if i < len(heights) else None)


dump('grayscale', gray)

bin_img = gray.point(lambda p: 255 if p > 150 else 0)
dump('binarize', bin_img)
