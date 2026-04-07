#!/usr/bin/env python3
"""
autoclicker.py - minimal OCR-based autoclicker.

Usage examples:
  python autoclicker.py --target "Click" --region 100 100 800 400

Notes:
- Install Tesseract OCR separately and ensure it's on PATH or pass --tesseract-cmd.
- This tool automates clicks based on OCR; use responsibly and respect service rules.
"""

import argparse
import time
import re
import sys
import os
from threading import Event
from typing import Optional, Callable

from PIL import Image, ImageOps
import pytesseract
import mss
import pyautogui
import keyboard
import cv2
import numpy as np
import logging

pyautogui.FAILSAFE = True


def parse_args():
    p = argparse.ArgumentParser(description="OCR-based autoclicker")
    p.add_argument("--target", "-t", required=True,
                   help="Target word to trigger a click (case-insensitive).")
    p.add_argument("--regex", action="store_true",
                   help="Treat target as a regular expression.")
    p.add_argument("--region", "-r", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="Capture region: left top width height. Defaults to primary monitor.")
    p.add_argument("--interval", "-i", type=float, default=0.7,
                   help="Minimum seconds between clicks.")
    p.add_argument("--hotkey", default="\\",
                   help="Hotkey to toggle scanning (default: \\).")
    p.add_argument("--start", action="store_true",
                   help="Start scanning immediately without waiting for hotkey.")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't perform actual clicks; only log detections.")
    p.add_argument("--confidence", type=float, default=60.0,
                   help="Minimum OCR confidence (0-100) to accept a word.")
    p.add_argument("--tesseract-cmd", default=None,
                   help="Path to tesseract executable (optional).")
    p.add_argument("--binarize", action="store_true",
                   help="Apply simple binarization to improve OCR accuracy.")
    p.add_argument("--threshold", type=int, default=150,
                   help="Binarization threshold (0-255).")
    p.add_argument("--pause", type=float, default=0.05,
                   help="Sleep between loop iterations to reduce CPU.")
    p.add_argument("--debug", action="store_true",
                   help="Print OCR debug info each capture (for troubleshooting).")
    p.add_argument("--template", "-T", default=None,
                   help="Path to a template image file to use template-matching for clicks.")
    p.add_argument("--template-threshold", type=float, default=0.8,
                   help="Match threshold for template-matching (0-1, higher is stricter).")
    p.add_argument("--template-scales", nargs="+", type=float, default=[1.0],
                   help="List of scales to try for the template (e.g. 0.9 1.0 1.1).")
    p.add_argument("--edge-threshold", type=float, default=0.45,
                   help="Edge-based match threshold (0-1, lower is looser).")
    p.add_argument("--auto-verify", action="store_true",
                   help="Automatically send /verify <code> when a verification code is detected in the capture region.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level (default: INFO).")
    p.add_argument("--log-file", default=None,
                   help="Optional path to a logfile. If omitted logs go to stdout.")
    p.add_argument("--verify-pattern", default=r'Code:\s*([A-Za-z0-9]{3,12})',
                   help="Regex (with a capture group) to extract the code from OCR text. Default: 'Code:\s*([A-Za-z0-9]{3,12})'.")
    p.add_argument("--chat-click", nargs=2, type=int, metavar=("X", "Y"),
                   help="Optional coordinates to click to focus the Discord chat input before sending the verify command.")
    p.add_argument("--verify-cooldown", type=float, default=10.0,
                   help="Minimum seconds before sending the same verification code again.")
    return p.parse_args()


def configure_logging(level: str = "INFO", logfile: Optional[str] = None):
    """Configure module-level logging."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    handlers = []
    if logfile:
        handlers = [logging.FileHandler(logfile)]
    else:
        handlers = [logging.StreamHandler()]
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s:%(name)s: %(message)s", handlers=handlers)


def _log(msg: str, log_callback: Optional[Callable[[str], None]] = None, logger: Optional[logging.Logger] = None, level: str = "info"):
    """Unified logging helper: prefer `log_callback` if provided, else logger."""
    if log_callback:
        try:
            log_callback(msg)
            return
        except Exception:
            pass
    if logger is None:
        logger = logging.getLogger('autoclicker')
    getattr(logger, level)(msg)


def capture_and_preprocess(sct, region, binarize: bool, threshold: int):
    sct_img = sct.grab(region)
    img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray_cv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if binarize:
        _, gray_cv_bin = cv2.threshold(gray_cv, threshold, 255, cv2.THRESH_BINARY)
        gray_for_ocr = Image.fromarray(gray_cv_bin)
    else:
        gray_for_ocr = ImageOps.grayscale(img)
    return img, frame_bgr, gray_cv, gray_for_ocr


def template_match(gray_cv, template_img, template_scales=None, template_threshold: float = 0.8, edge_threshold: float = 0.45, debug: bool = False, log_callback: Optional[Callable[[str], None]] = None, logger: Optional[logging.Logger] = None):
    if template_img is None:
        return None
    scales = template_scales if template_scales else [1.0]
    best_score = -1.0
    best_loc = None
    best_size = None
    best_edge_score = -1.0
    best_edge_loc = None
    best_edge_size = None
    try:
        edges_cv = cv2.Canny(gray_cv, 50, 150)
    except Exception:
        edges_cv = None

    for sc in scales:
        try:
            if sc == 1.0:
                tmpl = template_img
            else:
                tw = max(1, int(template_img.shape[1] * sc))
                th = max(1, int(template_img.shape[0] * sc))
                interp = cv2.INTER_AREA if sc < 1.0 else cv2.INTER_CUBIC
                tmpl = cv2.resize(template_img, (tw, th), interpolation=interp)
        except Exception:
            continue
        th, tw = tmpl.shape[:2]
        if th >= gray_cv.shape[0] or tw >= gray_cv.shape[1]:
            continue

        try:
            res = cv2.matchTemplate(gray_cv, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except Exception:
            max_val = -1.0
            max_loc = (0, 0)

        edge_max_val = -1.0
        edge_max_loc = (0, 0)
        if edges_cv is not None:
            try:
                tmpl_edges = cv2.Canny(tmpl, 50, 150)
                res_e = cv2.matchTemplate(edges_cv, tmpl_edges, cv2.TM_CCOEFF_NORMED)
                _, edge_max_val, _, edge_max_loc = cv2.minMaxLoc(res_e)
            except Exception:
                edge_max_val = -1.0

        if debug:
            _log(f"TEMPLATE scale={sc} score={max_val:.3f} at {max_loc} | EDGE score={edge_max_val:.3f} at {edge_max_loc}", log_callback, logger, level='debug')

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_size = (tw, th)
        if edge_max_val > best_edge_score:
            best_edge_score = edge_max_val
            best_edge_loc = edge_max_loc
            best_edge_size = (tw, th)

    if best_score >= template_threshold:
        return {'type': 'template', 'score': best_score, 'loc': best_loc, 'size': best_size}
    if best_edge_score >= edge_threshold:
        return {'type': 'edge', 'score': best_edge_score, 'loc': best_edge_loc, 'size': best_edge_size}
    return None


def ocr_to_words(gray_for_ocr):
    data = pytesseract.image_to_data(gray_for_ocr, output_type=pytesseract.Output.DICT, config='--psm 6')
    n = len(data.get("text", []))

    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])
    line_nums = data.get("line_num", [0] * n)

    words = []
    for i in range(n):
        try:
            txt = str(texts[i]).strip()
        except Exception:
            txt = ""
        if not txt:
            continue
        try:
            conf_val = float(confs[i])
        except Exception:
            conf_val = -1.0
        try:
            l = int(lefts[i])
            t = int(tops[i])
            w0 = int(widths[i])
            h0 = int(heights[i])
        except Exception:
            continue
        ln = line_nums[i] if i < len(line_nums) else 0
        words.append({"text": txt, "conf": conf_val, "left": l, "top": t, "width": w0, "height": h0, "line_num": ln})
    return words


def handle_auto_verify(words, gray_for_ocr, verify_pattern, verify_cooldown, chat_click, dry_run, log_callback: Optional[Callable[[str], None]] = None, logger: Optional[logging.Logger] = None):
    """Look for verification codes in OCR output and send /verify <code> if found.

    Keeps simple per-process cooldown using attributes on this function (previous behavior).
    """
    try:
        ocr_block = "\n".join([w.get('text', '') for w in words])
        if not ocr_block.strip():
            ocr_block = pytesseract.image_to_string(gray_for_ocr)
        m = None
        if verify_pattern:
            try:
                m = re.search(verify_pattern, ocr_block, flags=re.IGNORECASE)
            except Exception:
                m = None
        if not m:
            for alt in (r'/verify\s*([A-Za-z0-9]{3,12})', r'Code:\s*([A-Za-z0-9]{3,12})'):
                try:
                    m = re.search(alt, ocr_block, flags=re.IGNORECASE)
                except Exception:
                    m = None
                if m:
                    break
        if m:
            code = m.group(1)
            if not hasattr(handle_auto_verify, '_last_verified_code'):
                handle_auto_verify._last_verified_code = None
                handle_auto_verify._last_verify_time = 0.0
            now = time.time()
            if code and (code != handle_auto_verify._last_verified_code or (now - handle_auto_verify._last_verify_time) > verify_cooldown):
                if chat_click and len(chat_click) == 2:
                    try:
                        pyautogui.click(chat_click[0], chat_click[1])
                        time.sleep(0.12)
                    except Exception:
                        pass
                cmd = f"/verify {code}"
                try:
                    if not dry_run:
                        pyautogui.write(cmd, interval=0.02)
                        pyautogui.press('enter')
                    handle_auto_verify._last_verified_code = code
                    handle_auto_verify._last_verify_time = now
                    msg = f"[{time.strftime('%H:%M:%S')}] Sent '{cmd}'"
                    _log(msg, log_callback, logger)
                except Exception as e:
                    _log(f"Verify send failed: {e}", log_callback, logger, level='error')
    except Exception:
        # keep auto-verify best-effort and never crash scanner
        pass

def run_scanner(target,
                region,
                regex,
                interval,
                confidence,
                binarize,
                threshold,
                pause,
                tesseract_cmd,
                running_event: Event,
                stop_event: Event,
                dry_run: bool = False,
                debug: bool = False,
                template_img = None,
                template_threshold: float = 0.8,
                template_scales: Optional[list] = None,
                edge_threshold: float = 0.45,
                auto_verify: bool = False,
                verify_pattern: Optional[str] = None,
                chat_click: Optional[list] = None,
                verify_cooldown: float = 10.0,
                log_callback: Optional[Callable[[str], None]] = None):
    """Run the OCR capture->match->click loop until `stop_event` is set.

    - `running_event` controls whether scanning is active (set = scanning).
    - `log_callback` is called with a string for informational messages.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    sct = mss.mss()
    if not region:
        mon = sct.monitors[1]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
    else:
        if isinstance(region, (list, tuple)) and len(region) == 4:
            left, top, width, height = region
            region = {"left": left, "top": top, "width": width, "height": height}

    pattern = re.compile(target, flags=re.IGNORECASE) if regex else None
    target_lower = target.lower() if not regex else None

    last_click = 0.0
    logger = logging.getLogger('autoclicker')
    try:
        while not (stop_event and stop_event.is_set()):
            if running_event and running_event.is_set():
                # capture + preprocess
                img, frame_bgr, gray_cv, gray_for_ocr = capture_and_preprocess(sct, region, binarize, threshold)

                # template matching (fast path)
                matched_any = False
                if template_img is not None:
                    match = template_match(gray_cv, template_img, template_scales, template_threshold, edge_threshold, debug, log_callback, logger)
                    if match:
                        width, height = match['size']
                        top_left = match['loc']
                        cx = region["left"] + top_left[0] + width // 2
                        cy = region["top"] + top_left[1] + height // 2
                        if (time.time() - last_click) >= interval:
                            try:
                                if not dry_run:
                                    pyautogui.click(cx, cy)
                                last_click = time.time()
                                _log(f"[{time.strftime('%H:%M:%S')}] Template-clicked at ({cx},{cy}) score={match['score']:.3f} type={match['type']}", log_callback, logger)
                            except Exception as e:
                                _log(f"Click failed: {e}", log_callback, logger, level='error')
                        matched_any = True

                if matched_any:
                    time.sleep(pause)
                    continue

                # OCR fallback
                words = ocr_to_words(gray_for_ocr)

                if debug:
                    _log(f"DEBUG: {len(words)} words detected", log_callback, logger, level='debug')
                    for w in words:
                        _log(f"  '{w['text']}' conf={w['conf']} bbox=({w['left']},{w['top']},{w['width']},{w['height']}) line={w['line_num']}", log_callback, logger, level='debug')

                # auto-verify now that OCR words exist
                if auto_verify:
                    handle_auto_verify(words, gray_for_ocr, verify_pattern, verify_cooldown, chat_click, dry_run, log_callback, logger)

                matched_any = False

                # multi-word phrase matching
                if (not regex) and (" " in target):
                    lines = {}
                    for w in words:
                        ln = w.get("line_num", 0)
                        lines.setdefault(ln, []).append(w)

                    for ln, wlist in lines.items():
                        wlist.sort(key=lambda x: x["left"])
                        L = len(wlist)
                        for s in range(L):
                            for e in range(s, L):
                                seq = wlist[s:e+1]
                                joined = " ".join([x["text"] for x in seq]).strip()
                                seq_conf = min([x["conf"] for x in seq]) if seq else -1.0
                                if seq_conf < confidence:
                                    continue
                                if joined.lower() == target_lower:
                                    left_min = min(x["left"] for x in seq)
                                    top_min = min(x["top"] for x in seq)
                                    right_max = max(x["left"] + x["width"] for x in seq)
                                    bottom_max = max(x["top"] + x["height"] for x in seq)
                                    cx = region["left"] + left_min + (right_max - left_min) // 2
                                    cy = region["top"] + top_min + (bottom_max - top_min) // 2
                                    if (time.time() - last_click) >= interval:
                                        try:
                                            if not dry_run:
                                                pyautogui.click(cx, cy)
                                            last_click = time.time()
                                            _log(f"[{time.strftime('%H:%M:%S')}] Clicked on '{joined}' at ({cx},{cy}) conf={seq_conf}", log_callback, logger)
                                        except Exception as e:
                                            _log(f"Click failed: {e}", log_callback, logger, level='error')
                                    matched_any = True
                                    break
                            if matched_any:
                                break
                        if matched_any:
                            break

                # single-word/regex fallback
                if not matched_any:
                    for w in words:
                        if w["conf"] < confidence:
                            continue
                        text = w["text"]
                        matched = False
                        if regex:
                            if pattern.search(text):
                                matched = True
                        else:
                            if text.lower() == target_lower:
                                matched = True
                        if matched and (time.time() - last_click) >= interval:
                            cx = region["left"] + w["left"] + w["width"] // 2
                            cy = region["top"] + w["top"] + w["height"] // 2
                            try:
                                if not dry_run:
                                    pyautogui.click(cx, cy)
                                last_click = time.time()
                                _log(f"[{time.strftime('%H:%M:%S')}] Clicked on '{text}' at ({cx},{cy}) conf={w['conf']}", log_callback, logger)
                            except Exception as e:
                                _log(f"Click failed: {e}", log_callback, logger, level='error')
                            break

            time.sleep(pause)
    except KeyboardInterrupt:
        # allow clean exit on Ctrl+C
        pass


def main():
    args = parse_args()
    # When running as a PyInstaller bundle, auto-start scanning and use embedded data
    if getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS'):
        meipass = getattr(sys, '_MEIPASS', None)
        # Auto-start scanning when double-clicked (no args)
        if not args.start:
            args.start = True
        # Ensure live clicking when double-clicked (disable dry-run)
        args.dry_run = False
        # When bundled, enable auto-verify by default
        if not getattr(args, 'auto_verify', False):
            args.auto_verify = True
        # If no template path provided, use the embedded template from the bundle
        if not getattr(args, 'template', None) and meipass:
            bundled_template = os.path.join(meipass, 'template_fish.png')
            if os.path.exists(bundled_template):
                args.template = bundled_template

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    # configure logging
    configure_logging(getattr(args, 'log_level', 'INFO'), getattr(args, 'log_file', None))
    logger = logging.getLogger('autoclicker')
    _log("OCR autoclicker starting.", logger=logger)
    _log(f"Press '{args.hotkey}' to toggle scanning. Press ESC to exit.", logger=logger)

    running_event = Event()
    stop_event = Event()

    if args.start:
        running_event.set()
        _log("Scanning: ON", logger=logger)

    def toggle():
        if running_event.is_set():
            running_event.clear()
            _log("Scanning: OFF", logger=logger)
        else:
            running_event.set()
            _log("Scanning: ON", logger=logger)

    keyboard.add_hotkey(args.hotkey, toggle)
    keyboard.add_hotkey("esc", lambda: (stop_event.set(), sys.exit(0)))

    sct = mss.mss()
    if args.region:
        left, top, width, height = args.region
        region = {"left": left, "top": top, "width": width, "height": height}
    else:
        mon = sct.monitors[1]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}

    # Load template image if requested
    template_img = None
    if getattr(args, 'template', None):
        try:
            template_img = cv2.imread(args.template, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                _log(f"Template not found or unreadable: {args.template}", logger=logger, level='warning')
            else:
                _log(f"Loaded template: {args.template} size={template_img.shape[::-1]}", logger=logger)
        except Exception as e:
            _log(f"Failed to load template: {e}", logger=logger, level='error')

    try:
        run_scanner(args.target,
            region,
            args.regex,
            args.interval,
            args.confidence,
            args.binarize,
            args.threshold,
            args.pause,
            args.tesseract_cmd,
            running_event,
            stop_event,
            args.dry_run,
            args.debug,
            template_img,
            args.template_threshold,
            args.template_scales,
            args.edge_threshold,
            args.auto_verify,
            args.verify_pattern,
            args.chat_click,
            args.verify_cooldown,
            log_callback=None)
    except pytesseract.pytesseract.TesseractNotFoundError:
        _log("Tesseract not found. Install Tesseract OCR or pass its path with --tesseract-cmd.", logger=logger, level='error')
        _log("Windows installer: https://github.com/UB-Mannheim/tesseract/wiki", logger=logger, level='error')
    except KeyboardInterrupt:
        _log("Exiting...", logger=logger)


if __name__ == '__main__':
    main()
