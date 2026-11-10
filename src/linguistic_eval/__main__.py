"""Make python -m linguistic_eval use the independent evaluation CLI."""

from .cli import main


if __name__ == '__main__':
    raise SystemExit(main())
