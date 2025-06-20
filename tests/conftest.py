import sys
import os
import pytest

# Add app/ to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # Optionally clear/prepare memory or logs before tests
    pass 