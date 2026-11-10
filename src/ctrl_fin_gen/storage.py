"""Shared file operations for experiment records and generation checkpoints.

Structured snapshots (manifest/state/template) use JSON; incremental requests,
results, attempts, and streamed events use JSONL, one JSON record per line.
Writes retain Finnish characters. Snapshot replacement and flushed incremental
writes reduce lost/corrupt checkpoints during interruption; repair_jsonl handles
an unfinished last record without hiding corruption in earlier records.

generation.py owns the submission lifecycle and calls submission_lock before
mutating existing submissions. These helpers know nothing about provider APIs,
prompt layout, or linguistic checks.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .models import SCHEMA_VERSION


def timestamp():
    """Return an ISO-format UTC timestamp for saved records."""
    return datetime.now(timezone.utc).isoformat()


def content_hash(path):
    """Return the file's SHA-256 hex digest for source/provenance consistency checks."""
    return sha256(Path(path).read_bytes()).hexdigest()


def named_folder(parent, name):
    """Return parent/name after checking it is a single nonempty folder name.

    Reject path separators, '.'/'..', and surrounding whitespace. This does not
    create a folder or check whether it already exists; callers handle that.
    """
    if not name or name in ('.', '..') or name != name.strip() or '/' in name or '\\' in name:
        raise ValueError('Use a nonempty folder name without path separators.')
    return Path(parent) / name


def read_json(path):
    """Read UTF-8 JSON and return its decoded value; perform no schema validation."""
    with Path(path).open(encoding='utf-8') as source:
        return json.load(source)


def write_json(path, value):
    """Replace a JSON snapshot using a flushed temporary sibling file.

    The parent must exist. Reject NaN/infinity, preserve Unicode, and replace the
    destination only after serialization and fsync finish. This does not lock;
    callers updating an existing submission must hold submission_lock.
    """
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def append_jsonl(path, value):
    """Append one JSON record, flush it, and fsync before returning.

    The parent must exist; the file is created if absent. The caller handles
    locking and any unfinished trailing line before appending after a crash.
    """
    with Path(path).open('a', encoding='utf-8') as output:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        output.flush()
        os.fsync(output.fileno())


def read_jsonl(path):
    """Return decoded records, or [] if the file has not been created yet.

    Ignore only an invalid final line without a newline, which may be a partial
    crash write. Malformed complete/earlier lines raise ValueError. This read
    does not modify the file; repair_jsonl fixes its tail before future writes.
    """
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
    """Repair an unfinished final JSONL line in place; return None.

    Back up invalid trailing bytes to a sibling .partial file, then remove them
    from the journal. A valid final record lacking a newline only needs the
    newline added. Missing files are allowed. Call while holding the submission
    lock, before appending resumed results/events; earlier corruption is not fixed.
    """
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
    """Hold a submission's exclusive .lock file for the duration of a with block.

    Store the writer PID and remove the lock on normal/error exit. An existing
    lock raises ValueError rather than allowing two writers. A hard process kill
    may leave the file behind; it must only be removed after checking no writer
    remains. There is intentionally no automatic stale-lock detection here.
    """
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


def generation_folder(root, experiment, name):
    """Resolve a submission without creating or overwriting anything."""
    return named_folder(named_folder(Path(root) / 'experiments', experiment) / 'generations', name)


def load_manifest(folder):
    """Reject legacy/native formats before locks, repairs, or service operations."""
    manifest = read_json(folder / 'manifest.json')
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported submission format. Leave older runs intact and create a fresh run; static planning examples cannot be resumed.')
    if content_hash(folder / 'requests.jsonl') != manifest.get('request_plan_sha256'):
        raise ValueError('Frozen request plan has changed or is incomplete. Create a fresh run.')
    return manifest
