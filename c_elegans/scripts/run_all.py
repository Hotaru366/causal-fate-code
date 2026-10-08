#!/usr/bin/env python3
"""Run the original convergence, selection and evaluation protocol."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from celegans_dtpr_v4.experiment import main
if __name__ == "__main__":
    main()
