"""
Handwriting Engine CLI — backwards-compatible entry point.
The actual CLI lives in handwriting_engine/cli.py for proper pip-install support.
"""

from handwriting_engine.cli import cli

if __name__ == "__main__":
    cli()
