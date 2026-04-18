#!/usr/bin/env python
"""
Wrapper script to run Streamlit app
Use this if Databricks Apps doesn't recognize app.yaml
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port=8080",
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ])
