"""Helpers for packaged templates.

Expose a small API to list available templates and resolve their file paths.
"""
from pathlib import Path
from typing import List, Optional

TEMPLATES_DIR = Path(__file__).parent

def list_templates() -> List[str]:
    """Return a list of template filenames included with the package."""
    try:
        return [p.name for p in TEMPLATES_DIR.iterdir() if p.is_file()]
    except Exception:
        return []


def template_path(name: str = "template_fish.png") -> Optional[str]:
    """Return the filesystem path to a packaged template, or None if missing."""
    p = TEMPLATES_DIR / name
    return str(p) if p.exists() else None


__all__ = ["TEMPLATES_DIR", "list_templates", "template_path"]
