import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

# This is the entry point for Vercel
# The 'app' variable is what Vercel will use as the ASGI application
__all__ = ["app"] 