import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Execute the main streamlit app
with open(os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py"), encoding="utf-8") as f:
    code = f.read()
exec(code)
