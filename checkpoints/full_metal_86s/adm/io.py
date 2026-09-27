"""Fail the run when an image codec cannot write an output (including full disks)."""
from pathlib import Path
import cv2


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Image export failed: {Path(path)}')
