#!/usr/bin/python3

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / '/.venv/bin/python'))

from server import app
application = app