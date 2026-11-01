"""Discover or test all Bedrock model targets in the configured region."""

import sys
from pathlib import Path

# Direct script execution puts scripts/aws, rather than the checkout, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.bedrock_utils.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
