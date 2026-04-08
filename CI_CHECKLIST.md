# CI Checklist

- [ ] Run syntax checks: `python -m py_compile $(git ls-files '*.py')`
- [ ] Run unit tests with `pytest` (add `tests/` content)
- [ ] Add a separate job to install system deps (Tesseract OCR, OpenCV native libs)
- [ ] Add integration jobs for GUI/system-dependent tests (Windows/macOS)
- [ ] Cache pip dependencies to speed up CI
- [ ] Add a job to build distribution artifacts (PyInstaller) and verify
- [ ] Optionally: nightly run for long-running integration tests

Notes:
- Tesseract and GUI-related tests often require additional system packages or a headless X server.
- Keep unit tests small and isolated; run heavy tests in dedicated jobs.
