"""Small file helpers for experiment records and durable checkpoints."""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def content_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def named_folder(parent, name):
    if not name or name in ('.', '..') or name != name.strip() or '/' in name or '\\' in name:
        raise ValueError('Use a nonempty folder name without path separators.')
    return Path(parent) / name


def read_json(path):
    with Path(path).open(encoding='utf-8') as source:
        return json.load(source)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as output:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        output.flush()
        os.fsync(output.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b'\n'):
                break  # A process may have stopped during its final write.
            raise ValueError(f'Invalid JSONL record in {path}, line {index + 1}.')
    return rows


def repair_jsonl(path):
    """Repair only an unfinished final line, preserving its bytes separately."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b'\n'):
        return
    tail_start = data.rfind(b'\n') + 1
    try:
        json.loads(data[tail_start:])
    except (ValueError, UnicodeDecodeError):
        with path.with_name(path.name + '.partial').open('ab') as backup:
            backup.write(data[tail_start:] + b'\n')
        path.write_bytes(data[:tail_start])
    else:
        with path.open('ab') as output:
            output.write(b'\n')


@contextmanager
def submission_lock(folder):
    lock = Path(folder) / '.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f'Submission is locked: {lock}. After a crashed process, remove this file only when no writer is running.') from None
    try:
        with os.fdopen(descriptor, 'w') as output:
            output.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()
