"""Convenience entry point; see python -m adm.pipeline --help."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adm.pipeline import main
if __name__ == "__main__": main()
