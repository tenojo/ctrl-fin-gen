"""Command-line arguments and a thin layer over the package functions."""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAIError

from .generation import collect_results, get_status, resume_generation, start_generation
from .prepare import create_experiment, load_values
from .prompts import build_messages


def gen_parser() -> argparse.ArgumentParser:
    """Define commands without reading files or calling a generation provider."""
    parser = argparse.ArgumentParser(
        prog="ctrl-fin-gen",
        description="Create controlled-generation experiments and manage submissions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Create an experiment and choose its linguistic conditions.
    create = commands.add_parser("create", help="Create a new experiment.")
    create.add_argument("experiment", help="New folder name under experiments/.")
    create.add_argument(
        "--template", type=Path, required=True,
        help="System-template name, folder path, or template JSON file.",
    )
    create.add_argument(
        "--copy-from", help="Copy values from an experiment using the same system template."
    )
    create.add_argument(
        "--condition", type=Path,
        help="Values for controls exposed by the selected system template.",
    )
    create.add_argument(
        "--preview", action="store_true",
        help="Preview the proposed experiment without creating its folder.",
    )

    # The prefix, theme/task, and reminder occupy separate prompt sections.
    prompts = create.add_argument_group("Prompt text")
    system = prompts.add_mutually_exclusive_group()
    system.add_argument("--system-prompt", help="Separate system instructions.")
    system.add_argument("--system-prompt-file", type=Path, help="System prompt file.")

    prefix = prompts.add_mutually_exclusive_group()
    prefix.add_argument("--user-prefix", help="General user prompt prefix.")
    prefix.add_argument("--user-prefix-file", type=Path, help="User prefix file.")

    theme = prompts.add_mutually_exclusive_group()
    theme.add_argument("--prompt", help="Free-form style, theme, or task instructions.")
    theme.add_argument("--prompt-file", type=Path, help="Style/theme/task prompt file.")

    reminder = prompts.add_mutually_exclusive_group()
    reminder.add_argument("--user-reminder", help="Closing user prompt reminder.")
    reminder.add_argument("--user-reminder-file", type=Path, help="User reminder file.")

    # None means no CLI override; template/condition defaults are resolved later.
    length = create.add_argument_group("Length and target audience")
    length.add_argument(
        "--sentences", type=int, nargs=2, metavar=("MIN", "MAX"),
        help="Sentence-count bounds; use 5 5 for exactly five sentences.",
    )
    length.add_argument(
        "--words", type=int, nargs=2, metavar=("MIN", "MAX"),
        help="Word-count bounds; use equal bounds for an exact count.",
    )
    length.add_argument("--max-word-length", type=int, help="Maximum word length.")
    length.add_argument(
        "--word-length-unit", choices=("letters", "characters"),
        help="Unit used for the maximum word length.",
    )
    audience = length.add_mutually_exclusive_group()
    audience.add_argument(
        "--audience", help="Target audience, including education level or discipline."
    )
    audience.add_argument(
        "--audience-file", type=Path, help="File describing the target audience."
    )

    # Empty inline lists explicitly request zero items.
    vocabulary = create.add_argument_group("Vocabulary selection")
    source = vocabulary.add_mutually_exclusive_group()
    source.add_argument("--lemmas", nargs="*", metavar="LEMMA", help="Lemmas to use.")
    source.add_argument(
        "--lemmas-file", type=Path, help="Chosen lemmas, one per line."
    )
    source.add_argument(
        "--vocabulary-pool", type=Path,
        help="Vocabulary pool or import specification to sample from.",
    )
    vocabulary.add_argument(
        "--sample-lemmas", type=int, help="Number of lemmas to sample from the pool."
    )
    vocabulary.add_argument("--seed", type=int, help="Vocabulary sampling seed.")
    vocabulary.add_argument(
        "--selection-scope", choices=("condition", "request"),
        help="Sample once per condition or separately for each request.",
    )

    controls = create.add_argument_group("Catalogue presence controls")
    controls.add_argument(
        "--linguistic-controls", nargs="*", metavar="CODE",
        help="Morpho-lexical presence controls, such as Case=Gen.",
    )
    controls.add_argument(
        "--syntax-controls", nargs="*", metavar="CODE",
        help="Syntax-tree presence controls, such as acl:relcl.",
    )
    controls.add_argument(
        "--pos-controls", nargs="*", metavar="CODE",
        help="Part-of-speech presence controls, such as NOUN.",
    )

    # Each new submission gets its own named generation folder.
    generate = commands.add_parser("generate", help="Start or resume a generation.")
    generate.add_argument("experiment", help="Existing experiment folder name.")
    target = generate.add_mutually_exclusive_group()
    target.add_argument(
        "--name", help="New generation folder name; ask for one when omitted."
    )
    target.add_argument(
        "--resume", metavar="GENERATION",
        help="Continue an interrupted generation using its saved settings.",
    )
    generate.add_argument(
        "--retry-unknown", action="store_true",
        help="With --resume, explicitly retry a request whose remote outcome is unknown.",
    )

    settings = generate.add_argument_group(
        "Generation settings",
        "New submissions require backend, model, max-length, and samples. "
        "Resume uses saved settings instead.",
    )
    settings.add_argument("--backend", choices=("openai", "google", "vllm"))
    settings.add_argument("--model", help="Model name or identifier.")
    settings.add_argument(
        "--max-length", "--max-output-tokens", dest="max_output_tokens",
        type=int, metavar="TOKENS", help="Maximum model output budget, in tokens.",
    )
    settings.add_argument(
        "--samples", type=int, help="Number of separately identified text-generation requests."
    )
    settings.add_argument(
        "--reasoning-effort", help="Reasoning effort supported by the chosen model."
    )
    settings.add_argument(
        "--base-url", help="Custom API base URL, for example a running vLLM server."
    )
    settings.add_argument(
        "--google-api", choices=("developer", "cloud"), help="Google API route."
    )
    settings.add_argument("--google-project", help="Google Cloud project identifier.")
    settings.add_argument("--google-location", help="Google Cloud location.")
    processing = settings.add_mutually_exclusive_group()
    processing.add_argument("--stream", action="store_true", help="Stream immediate responses.")
    processing.add_argument("--batch", action="store_true", help="Submit a provider batch.")

    # These commands use the provider/job information saved in the submission.
    status = commands.add_parser("status", help="Check the status of a generation.")
    status.add_argument("experiment", help="Existing experiment folder name.")
    status.add_argument("generation", help="Existing generation folder name.")

    collect = commands.add_parser("collect", help="Fetch and format generation results.")
    collect.add_argument("experiment", help="Existing experiment folder name.")
    collect.add_argument("generation", help="Existing generation folder name.")

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse arguments and check new-submission versus resume options."""
    parser = gen_parser()
    args = parser.parse_args(argv)
    if args.command != "generate":
        return args

    if args.resume is not None:
        overrides = (
            args.backend, args.model, args.max_output_tokens, args.samples,
            args.reasoning_effort, args.base_url, args.google_api,
            args.google_project, args.google_location,
        )
        if any(value is not None for value in overrides) or args.stream or args.batch:
            parser.error("--resume reuses saved settings; do not supply generation settings.")
    else:
        if args.retry_unknown:
            parser.error("--retry-unknown requires --resume.")
        required = (args.backend, args.model, args.max_output_tokens, args.samples)
        if any(value is None for value in required):
            parser.error("A new generation requires --backend, --model, --max-length, and --samples.")
        if args.max_output_tokens <= 0 or args.samples <= 0:
            parser.error("--max-length and --samples must be positive integers.")

    return args


def creation_values(args):
    """Apply explicit CLI overrides to optional JSON condition values."""
    values = load_values(args.condition) if args.condition else {}
    constraints = values.setdefault('constraints', {})
    for option, key in (('sentences', 'number_of_sentences'), ('words', 'number_of_words')):
        value = getattr(args, option)
        if value is not None:
            constraints.setdefault('length_constraints', {})[key] = {'min': value[0], 'max': value[1]}
    if args.max_word_length is not None:
        constraints.setdefault('length_constraints', {})['max_word_length'] = args.max_word_length
    if args.word_length_unit is not None:
        values['word_length_unit'] = args.word_length_unit

    for option, group in (
        ('linguistic_controls', 'linguistic_constraints'),
        ('syntax_controls', 'syntax_tree_constraints'),
        ('pos_controls', 'part_of_speech_constraints'),
    ):
        items = getattr(args, option)
        if items is not None:
            constraints[group] = {'selection': {'mode': 'inline', 'items': items}}

    selection = None
    if args.lemmas is not None:
        selection = {'mode': 'inline', 'items': args.lemmas}
    elif args.lemmas_file is not None:
        selection = {'mode': 'file', 'path': str(args.lemmas_file.resolve())}
    elif args.vocabulary_pool is not None:
        selection = {'mode': 'sample', 'pool': str(args.vocabulary_pool.resolve())}
    sampling_options = {'count': args.sample_lemmas, 'seed': args.seed, 'scope': args.selection_scope}
    if any(value is not None for value in sampling_options.values()):
        selection = selection or constraints.get('vocabulary_constraints', {}).get('selection')
        if not selection or selection.get('mode') != 'sample':
            raise ValueError('Sampling options require --vocabulary-pool or a sample selection in --condition.')
    if selection is not None:
        selection.update({key: value for key, value in sampling_options.items() if value is not None})
        constraints['vocabulary_constraints'] = {'selection': selection}

    for option, key in (
        ('system_prompt', 'system'), ('user_prefix', 'user_prefix'),
        ('user_reminder', 'user_reminder'), ('prompt', 'prompt'), ('audience', 'audience'),
    ):
        text = getattr(args, option)
        path = getattr(args, option + '_file')
        if path is not None:
            text = path.read_text(encoding='utf-8')
        if text is not None:
            if key == 'audience':
                values[key] = {'description': text}
            elif key == 'prompt':
                values[key] = text
            else:
                values.setdefault('prompts', {})[key] = text
    return values


def print_stream_text(text):
    print(text, end='', file=sys.stderr, flush=True)


def run_command(args, root):
    if args.command == 'create':
        prepared = create_experiment(
            root, args.experiment, args.template, creation_values(args), args.copy_from, args.preview,
        )
        if args.preview:
            return {'template': prepared.template, 'messages': build_messages(prepared)}
        return {'experiment': args.experiment, 'path': str(root / 'experiments' / args.experiment)}
    if args.command == 'generate':
        if args.resume:
            return resume_generation(
                root, args.experiment, args.resume, retry_unknown=args.retry_unknown,
                on_text=print_stream_text,
            )
        if args.backend != 'openai':
            raise ValueError('Only the OpenAI backend is implemented.')
        if any(value is not None for value in (args.google_api, args.google_project, args.google_location)):
            raise ValueError('Google options cannot be used with the OpenAI backend.')
        name = args.name if args.name is not None else input('New generation folder name: ').strip()
        settings = {
            'backend': args.backend, 'model': args.model,
            'max_output_tokens': args.max_output_tokens, 'samples': args.samples,
            'reasoning_effort': args.reasoning_effort, 'base_url': args.base_url,
            'stream': args.stream, 'batch': args.batch,
        }
        return start_generation(root, args.experiment, name, settings, on_text=print_stream_text)
    if args.command == 'status':
        return get_status(root, args.experiment, args.generation)
    return collect_results(root, args.experiment, args.generation)


def main(argv=None, root=None):
    """Run from the project/workspace folder; return a shell exit code."""
    args = parse_args(argv)
    root = Path(root).resolve() if root is not None else Path.cwd()
    try:
        result = run_command(args, root)
    except KeyboardInterrupt:
        print('\nInterrupted. Use generate EXPERIMENT --resume NAME to continue.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, EOFError, OpenAIError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
