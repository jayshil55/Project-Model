#!/usr/bin/env python3
"""
NutriCare ML -- Premium Gemini-Inspired Terminal UI Dashboard
=============================================================
Main entry point for the NutriCare interactive terminal application.

Usage:
    python app.py
"""

import sys
import os

# Ensure UTF-8 output encoding where supported on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tui.app import NutriCareApp


def main():
    """Launch the NutriCare TUI Dashboard."""
    app = NutriCareApp()
    app.run()


if __name__ == "__main__":
    main()
