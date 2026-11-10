"""Independent parser-based checks on saved generation artifacts.

parser_process selects completed texts and saves constraints plus CoNLL-U.
feature_extractor measures features; parser_checks compares frozen targets.
args/cli provide the three-stage interface; trankit_support owns the backend.
Generation never imports this package. Future orchestration, fluency evaluation,
filtering, and statistical analysis remain outside the generation module.
"""
