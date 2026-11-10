"""Make ``python -m ctrl_fin_gen`` call the same CLI as ``ctrl-fin-gen``.

All argument handling and command dispatch live in cli.py. The returned status
becomes the process exit code; this file adds no generation logic.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
