"""Finnish Trankit-large/TDT parsing through a reusable, isolated CPU worker.

No model or NLP dependency is imported in the caller's Python environment.
Use one TrankitParser context for many texts to load the model only once.
Instances are sequential, not thread-safe. Analysis is separate from generation.
"""

import json
import os
from pathlib import Path
import subprocess

from .trankit_resources import default_cache


class ParserError(RuntimeError):
    """Unavailable resources or failed analysis, distinct from a failed constraint."""


class TrankitParser:
    def __init__(self, *, python=None, cache_dir=None, threads=1):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError('threads must be a positive integer.')
        root = Path(__file__).resolve().parents[3]
        local_python = root / '.venv-trankit' / (
            'Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        self.python = str(python or os.environ.get('CTRL_FIN_GEN_TRANKIT_PYTHON') or local_python)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else default_cache()
        self.threads = threads
        self._process = None
        self.metadata = None

    def _receive(self):
        line = self._process.stdout.readline()
        if not line:
            raise ParserError('Trankit worker stopped unexpectedly; see its stderr output.')
        try:
            message = json.loads(line)
        except ValueError as error:
            raise ParserError('Invalid response from Trankit worker.') from error
        if 'error' in message:
            raise ParserError(message['error'])
        return message

    def start(self):
        if self._process is not None:
            return self
        if not (self.cache_dir / 'resources.json').is_file():
            raise ParserError('Trankit resources are missing. Run: python3.11 scripts/setup_trankit.py')
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'TRANSFORMERS_CACHE': str(self.cache_dir / 'xlm-roberta-large'),
            'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': str(self.threads),
            'MKL_NUM_THREADS': str(self.threads),
        })
        try:
            self._process = subprocess.Popen(
                [self.python, '-m', 'linguistic_eval.trankit_support._trankit_worker',
                 '--cache-dir', str(self.cache_dir), '--threads', str(self.threads)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                encoding='utf-8', env=env,
            )
            self.metadata = self._receive()['ready']
        except OSError as error:
            self.close()
            raise ParserError('Cannot launch Trankit. Run python3.11 scripts/setup_trankit.py '
                              'or set CTRL_FIN_GEN_TRANKIT_PYTHON.') from error
        except Exception:
            self.close()
            raise
        return self

    def parse(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Text must be a non-empty string.')
        self.start()
        try:
            self._process.stdin.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            self._process.stdin.flush()
            return self._receive()['result']
        except (OSError, KeyError) as error:
            raise ParserError('Trankit worker communication failed.') from error

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
