"""Entry point for hosts that look for main.py (pocketnook and similar). Serves on $PORT."""

from jev.web import app, serve  # noqa: F401  (app is exposed for `uvicorn main:app`)

if __name__ == "__main__":
    serve()
