"""
server/app.py — OpenEnv server entry point shim.
Imports the FastAPI app from the root app module.
No sys.path hacks needed: the Dockerfile sets WORKDIR /app and copies everything there.
"""

from app import app  # noqa: F401 — re-exported for OpenEnv server discovery


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the OpenEnv HTTP server via uvicorn."""
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
