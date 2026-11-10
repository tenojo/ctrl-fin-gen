"""Run with Python 3.11: create the isolated CPU parser and fetch pinned models."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-dir', type=Path, default=Path('.venv-trankit'))
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/trankit-large-tdt'))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error('Run this setup with Python 3.11; the main project can use a newer Python.')
    root = Path(__file__).resolve().parents[1]
    env_dir = args.env_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    python = env_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env_dir)
    worker_env = os.environ.copy()
    worker_env['PYTHONPATH'] = str(root / 'src')
    install_env = os.environ.copy()
    install_env.pop('PYTHONPATH', None)

    def run(*cmd):
        # Source-tree egg-info on PYTHONPATH would make pip check incorrectly
        # inspect the main project's provider requirements inside the worker.
        env = worker_env if cmd[1].startswith('linguistic_eval') else install_env
        subprocess.run([str(python), *cmd], check=True, cwd=root, env=env)
    if sys.platform == 'darwin':
        run('-m', 'pip', 'install', 'torch==2.0.1')
    else:
        run('-m', 'pip', 'install', 'torch==2.0.1+cpu',
            '--index-url', 'https://download.pytorch.org/whl/cpu')
    run('-m', 'pip', 'install', '-r', str(root / 'requirements/trankit.txt'))
    # Run the project's worker from source, without installing provider SDKs here.
    run('-m', 'pip', 'check')
    run('-m', 'linguistic_eval.trankit_support.trankit_resources', '--cache-dir', str(cache_dir))
    run('-m', 'linguistic_eval', 'parse-text', '--python', str(python), '--cache-dir', str(cache_dir),
        '--text', 'Koulussa lapset lukevat kirjoja. Opettaja auttaa heitä.')


if __name__ == '__main__':
    main()
