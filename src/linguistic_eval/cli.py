"""Thin CLI dispatch; generation never calls or imports this module."""

import json
import sys

from .args import parse_args


def run_command(args):
    if args.command == 'extract':
        from .feature_extractor import extract_folder
        return extract_folder(args.input, args.output_folder, only_new=args.only_new)
    if args.command == 'check':
        from .parser_checks import check_folder
        return check_folder(args.input, constraints_path=args.constraints,
                            features_folder=args.feature_analysis, output_file=args.output, only_new=args.only_new)
    from .trankit_support import TrankitParser
    # Construction is lazy: --only-new does not load a model when every parse exists.
    with TrankitParser(python=args.python, cache_dir=args.cache_dir, threads=args.threads) as parser:
        if args.command == 'parse':
            from .parser_process import parse_folder
            return parse_folder(args.input_folder, args.output_folder, parser=parser, only_new=args.only_new)
        text = args.text if args.text is not None else (
            args.input.read_text(encoding='utf-8') if args.input else sys.stdin.read())
        return parser.parse(text)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_command(args)
        if args.command == 'parse-text' and args.format == 'conllu':
            print(result['conllu'], end='')
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        summaries = result if isinstance(result, list) else [result]
        return int(any(row.get('errors', 0) for row in summaries))
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        print(f'Evaluation failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
