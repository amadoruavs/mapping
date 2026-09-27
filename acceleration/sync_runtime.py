"""Keep the two independently installable applications on identical runtime code."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
for target in [ROOT.parent/'ODM-Fedora/parallel_runtime',ROOT.parent/'MapRepair/parallel_runtime']:
    shutil.copytree(ROOT/'parallel_runtime',target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
