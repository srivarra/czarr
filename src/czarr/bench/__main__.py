"""``python -m czarr.bench`` -> the typer app (used by --isolate re-exec)."""

from .cli import app

if __name__ == "__main__":
    app()
