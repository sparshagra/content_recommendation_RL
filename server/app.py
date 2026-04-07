"""
Content Recommendation OpenEnv — Server Entry Point
====================================================
Required by openenv validate: server/app.py with main() function.
This module starts the FastAPI server via uvicorn.
"""

import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the OpenEnv HTTP server."""
    import uvicorn

    # Import the FastAPI app from root
    uvicorn.run("app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
