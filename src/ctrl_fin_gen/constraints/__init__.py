"""Helpers for preparing requested controls, before text generation.

length.py validates numeric choices against system-template bounds.
vocabulary.py imports lemma lists and freezes reproducible selections. Shared
feature definitions and Finnish wording live in conditions/linguistic_controls/;
prepare.py resolves which of those controls a system template exposes.

These modules do not measure or filter generated text. Parser-backed constraint
checks and other evaluation methods belong to the later evaluation stage.
"""
