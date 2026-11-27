#!/usr/bin/env python3
"""Compatibility wrapper — delegates to scripts/sync.py."""
import pathlib
import runpy

if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "sync.py"), run_name="__main__")
