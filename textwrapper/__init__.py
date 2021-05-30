# append path
import sys
import os

if not os.path.normpath(os.path.join(os.path.dirname(__file__), '..')) in sys.path:
    sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))