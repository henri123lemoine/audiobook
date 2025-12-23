"""Legacy entry point - use `uv run audiobook` instead.

Usage:
    uv run audiobook generate --book absalon --chapter 1
    uv run audiobook info --book absalon
    uv run audiobook --help
"""

from src.cli import cli

if __name__ == "__main__":
    cli()
