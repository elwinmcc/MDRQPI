import sys
import os

# Ensure the project root is on the path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dashboard.app import server  # noqa: F401 — Vercel WSGI entry point

# Vercel calls this module and looks for a WSGI-compatible `app` or `server`.
app = server
