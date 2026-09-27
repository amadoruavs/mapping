"""Fail the run when an image codec cannot write an output (including full disks)."""
from pathlib import Path
import json
import cv2
import numpy as np
from .storage import atomic_output, check_write


def write_image(path, image):
    path = Path(path)
    # Reserve worst-case uncompressed image size before codec allocation/writes.
    check_write(path, image.nbytes+image.shape[0]*8+1024**2)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'Image export failed: {Path(path)}')
    with atomic_output(path) as stream:
        stream.write(memoryview(encoded).cast('B'))


def write_json(path, value):
    data = json.dumps(value, indent=2).encode()
    with atomic_output(path) as stream:
        stream.write(data)


def write_array(path, value):
    with atomic_output(path) as stream:
        np.save(stream, value)


def write_npz(path, **values):
    with atomic_output(path) as stream:
        np.savez_compressed(stream, **values)
