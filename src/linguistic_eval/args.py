"""Arguments for the independent three-stage linguistic evaluator."""

import argparse
from pathlib import Path


def _parser_options(command):
    command.add_argument('--python', help='Python executable for the isolated Trankit worker.')
    command.add_argument('--cache-dir', type=Path)
    command.add_argument('--threads', type=int, default=1)


def gen_parser():
    parser = argparse.ArgumentParser(prog='linguistic-eval', description='Parse, extract, and check Finnish generation outputs.')
    commands = parser.add_subparsers(dest='command', required=True)
    text = commands.add_parser('parse-text', help='Parse a single text for inspection.')
    source = text.add_mutually_exclusive_group()
    source.add_argument('--text')
    source.add_argument('--input', type=Path, help='UTF-8 file; defaults to stdin.')
    text.add_argument('--format', choices=('json', 'conllu'), default='json')
    _parser_options(text)

    saved = commands.add_parser('parse', help='Parse completed outputs from an experiment or generation.')
    saved.add_argument('--input-folder', type=Path, required=True)
    saved.add_argument('--output-folder', type=Path, help='Output root for the selected experiment or generation.')
    _parser_options(saved)

    extract = commands.add_parser('extract', help='Extract features from every CoNLL-U in a folder.')
    extract.add_argument('--input', type=Path, required=True, help='The conllus folder.')
    extract.add_argument('--output-folder', type=Path, help='Defaults to sibling feature_analysis/.')

    check = commands.add_parser('check', help='Check one generation evaluation folder.')
    check.add_argument('--input', type=Path, required=True, help='The evaluation folder for generation Y.')
    check.add_argument('--constraints', type=Path, help='Defaults to Y/constraints.json.')
    check.add_argument('--feature-analysis', type=Path, help='Defaults to Y/feature_analysis/.')
    check.add_argument('--output', type=Path, help='Defaults to Y/constraint_satisfaction.jsonl.')
    for command in (saved, extract, check):
        command.add_argument('--only-new', action='store_true', help='Reuse unchanged existing entries; process new/changed inputs.')
    return parser


def parse_args(argv=None):
    parser = gen_parser()
    args = parser.parse_args(argv)
    if args.command in ('parse-text', 'parse') and args.threads < 1:
        parser.error('--threads must be a positive integer.')
    return args
