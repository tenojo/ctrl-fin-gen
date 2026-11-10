"""Trankit backend; model dependencies stay inside its isolated worker.

Other linguistic_eval modules consume the TextParser interface in parser_process,
not Trankit's native classes. Importing this package does not load a model.
"""

from importlib import import_module

__all__ = ['ParserError', 'TrankitParser']


def __getattr__(name):
    # Resource provisioning is also executable with python -m. Avoid importing
    # its module via nlp before runpy gets to execute it.
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module('.nlp', __name__), name)
    globals()[name] = value
    return value
