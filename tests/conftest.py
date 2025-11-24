# test/conftest.py
import sys
from pathlib import Path

# Add the test directory to sys.path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))
