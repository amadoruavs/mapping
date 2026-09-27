"""Enforce a per-project output cap while streaming writes into atomic files."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import os
import shutil


_active_budget = ContextVar('adm_output_budget', default=None)


@dataclass(frozen=True)
class OutputBudget:
    root: Path
    limit_bytes: int
    reserve_free_bytes: int = 512*1024**2

    def used_bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def check(self, path, additional_bytes):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Output write escaped the project directory')
        required = max(0, int(additional_bytes))
        used = self.used_bytes()
        if used+required > self.limit_bytes:
            raise OSError(f'Output budget exceeded: {used+required} bytes requested, '
                          f'{self.limit_bytes} byte limit. Keep a vector project or explicitly increase the export budget.')
        if shutil.disk_usage(self.root).free-required < self.reserve_free_bytes:
            raise OSError('Output would consume the reserved free disk space')


@contextmanager
def output_budget(root, limit_bytes, reserve_free_bytes=512*1024**2):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    budget = OutputBudget(root, int(limit_bytes), int(reserve_free_bytes))
    budget.check(root, 0)
    token = _active_budget.set(budget)
    try:
        yield budget
    finally:
        _active_budget.reset(token)


def check_write(path, size):
    budget = _active_budget.get()
    if budget is not None:
        budget.check(path, size)


class _GuardedWriter:
    def __init__(self, stream, path):
        self.stream = stream
        self.path = path
        self.length = 0

    def write(self, data):
        end = self.stream.tell()+len(data)
        check_write(self.path, max(0, end-self.length))
        written = self.stream.write(data)
        # Flush before the next quota check so filesystem accounting is current.
        self.stream.flush()
        self.length = max(self.length, self.stream.tell())
        return written

    def tell(self):
        return self.stream.tell()

    def seek(self, *args):
        return self.stream.seek(*args)

    def flush(self):
        return self.stream.flush()

    # numpy's archive writer treats anything without read() as a filesystem path.
    def read(self, *args):
        return self.stream.read(*args)

    def readable(self):
        return self.stream.readable()

    def writable(self):
        return True

    def seekable(self):
        return self.stream.seekable()


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.partial')
    check_write(temporary, 0)
    try:
        with temporary.open('wb') as stream:
            yield _GuardedWriter(stream, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
