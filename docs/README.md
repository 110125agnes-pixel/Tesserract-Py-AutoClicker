# OCR Autoclicker

Minimal OCR-based autoclicker that watches a screen region for a specific word and clicks it.

Prerequisites
- Python 3.8+
- Install Python packages:

```bash
pip install -r requirements.txt
```

- Install Tesseract OCR (separate installer). On Windows, install Tesseract and ensure `tesseract.exe` is on your PATH or pass `--tesseract-cmd`.

Quick usage

```bash
python autoclicker.py --target "ClickMe" --region 100 100 800 200 --interval 0.8
```

Options
- `--target` (required): Word to look for (case-insensitive)
- `--regex`: treat `--target` as a regular expression
- `--region x y w h`: capture region (left top width height)
- `--interval`: min seconds between clicks
- `--hotkey`: toggle scanning hotkey (default: `ctrl+shift+s`)
- `--confidence`: min OCR confidence (default 60)

Safety & notes
- This automates clicks; using automation to gain unfair advantage may violate a service's rules. Use responsibly.
- `pyautogui` has a failsafe: move the mouse to the top-left corner to abort.

Testing & CI
--------------

This repository includes a minimal test placeholder and a basic CI workflow.

- Run the test suite locally:

```bash
python -m pip install --upgrade pip pytest
pytest -q
```

- Quick syntax check (compile all Python files):

```bash
python -m py_compile $(git ls-files "*.py")
```

- The project includes a GitHub Actions workflow at `.github/workflows/ci.yml` which runs syntax checks and `pytest` on pushes and pull requests.

If you add tests, place them under the `tests/` directory and keep them small and isolated from GUI/system dependencies.
