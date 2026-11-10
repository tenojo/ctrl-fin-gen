"""Parse completed saved outputs using one backend, without importing generation."""

from copy import deepcopy
from pathlib import Path
from typing import Mapping, Protocol

from ._files import digest, read_json, read_jsonl, write_json, write_text

PARSING_VERSION = '2'


class TextParser(Protocol):
    def parse(self, text: str) -> Mapping[str, object]:
        """Return text, conllu, and parser provenance; raise on failure."""
        ...


def parse_generation(generation_dir, output_dir, *, parser: TextParser, only_new=False):
    """Write constraints.json and conllus/; return counts for this generation."""
    generation_dir, output_dir = Path(generation_dir).resolve(), Path(output_dir).resolve()
    if output_dir == generation_dir or generation_dir in output_dir.parents:
        raise ValueError('Evaluation output must be outside the source generation folder.')
    requests = read_jsonl(generation_dir / 'requests.jsonl')
    manifest = read_json(generation_dir / 'manifest.json')
    requests = {r.get('request_id', r.get('sample_id')): r for r in requests}
    results = read_jsonl(generation_dir / 'results.jsonl')
    entries, seen = [], set()
    for result in results:
        if result.get('status') != 'completed':
            continue
        request_id = result.get('request_id', result.get('sample_id'))
        attempt_id = result.get('attempt_id')
        if (not isinstance(request_id, str) or not request_id or request_id not in requests
                or not isinstance(attempt_id, str) or not attempt_id):
            raise ValueError(f'Cannot match completed result to a request/attempt in {generation_dir}.')
        text_id = digest(f'{request_id}\0{attempt_id}')[:24]
        if text_id in seen:
            raise ValueError(f'Duplicate completed attempt in {generation_dir}: {attempt_id}')
        seen.add(text_id)
        request = requests[request_id]
        # The older prototype stored its frozen conditions in the manifest.
        conditions = request.get('conditions', manifest.get('condition'))
        if not isinstance(conditions, dict) or 'constraints' not in conditions:
            raise ValueError(f'Missing frozen constraints for {request_id}.')
        constraints = deepcopy(conditions['constraints'])
        if 'vocabulary' in request:
            constraints['vocabulary_constraints']['items'] = request['vocabulary']
        text = result.get('text', '')
        if not isinstance(text, str):
            raise ValueError(f'Invalid text for {request_id}.')
        entries.append({
            'text_id': text_id, 'request_id': request_id, 'attempt_id': attempt_id,
            'configuration_id': result.get('configuration_id', request.get('configuration_id')),
            'repetition_index': result.get('repetition_index', request.get('repetition_index')),
            'text': text, 'text_sha256': digest(text), 'constraints': constraints,
            'word_length_unit': 'characters',
        })
    constraints_path = output_dir / 'constraints.json'
    previous = read_json(constraints_path) if constraints_path.exists() else {'texts': []}
    write_json(constraints_path, {
        'schema_version': '1.0', 'source_folder': str(generation_dir),
        'experiment': generation_dir.parent.parent.name, 'generation': generation_dir.name,
        'skipped_results': len(results) - len(entries), 'texts': entries,
    })
    conllus = output_dir / 'conllus'
    conllus.mkdir(parents=True, exist_ok=True)
    # Remove only files owned by a previously indexed, now absent completed text.
    for old in previous['texts']:
        old_id = old['text_id']
        if (old_id not in seen and len(old_id) == 24
                and all(c in '0123456789abcdef' for c in old_id)):
            for suffix in ('.conllu', '.json'):
                (conllus / (old_id + suffix)).unlink(missing_ok=True)
    counts = {'output_folder': str(output_dir), 'parsed': 0, 'reused': 0, 'errors': 0,
              'skipped_results': len(results) - len(entries)}
    for entry in entries:
        path = conllus / (entry['text_id'] + '.conllu')
        metadata_path = path.with_suffix('.json')
        old = read_json(metadata_path) if only_new and metadata_path.exists() else {}
        if (only_new and path.exists() and old.get('status') == 'parsed'
                and old.get('parsing_version') == PARSING_VERSION
                and old.get('text_sha256') == entry['text_sha256']
                and old.get('conllu_sha256') == digest(path.read_text(encoding='utf-8'))):
            counts['reused'] += 1
            continue
        metadata = {'text_id': entry['text_id'], 'text_sha256': entry['text_sha256'],
                    'parsing_version': PARSING_VERSION}
        try:
            if not entry['text'].strip():
                raise ValueError('Completed output has no text.')
            parsed = parser.parse(entry['text'])
            conllu = parsed['conllu']
            if not isinstance(conllu, str) or not conllu.strip():
                raise ValueError('Parser returned no CoNLL-U.')
            write_text(path, conllu)
            metadata.update(status='parsed', conllu_sha256=digest(conllu), parser=parsed.get('parser', {}))
            counts['parsed'] += 1
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            path.unlink(missing_ok=True)
            metadata.update(status='error', error=str(error))
            counts['errors'] += 1
        write_json(metadata_path, metadata)
    return counts


def parse_experiment(experiment_dir, output_dir, *, parser: TextParser, only_new=False):
    """Parse each generation using the same parser; put each under output_dir/Y."""
    generations = Path(experiment_dir) / 'generations'
    if not generations.is_dir():
        raise ValueError(f'No generations folder: {generations}')
    return [parse_generation(folder, Path(output_dir) / folder.name, parser=parser, only_new=only_new)
            for folder in sorted(generations.iterdir()) if folder.is_dir() and (folder / 'results.jsonl').is_file()]


def parse_folder(input_folder, output_folder=None, *, parser: TextParser, only_new=False):
    """Detect experiment vs generation from the experiment's template folder."""
    folder = Path(input_folder).resolve()
    if (folder / 'template/template.json').is_file():
        experiment, whole_experiment = folder, True
    elif (folder.parent.parent / 'template/template.json').is_file() and folder.parent.name == 'generations':
        experiment, whole_experiment = folder.parent.parent, False
    else:
        raise ValueError('Supply an experiment with template/template.json, or one of its generations.')
    output = (Path(output_folder).resolve() if output_folder is not None else
              experiment.parent.parent / 'evaluations/deterministic_checks' / experiment.name)
    if not whole_experiment and output_folder is None:
        output /= folder.name
    if output == experiment or experiment in output.parents:
        raise ValueError('Evaluation output must be outside the source experiment folder.')
    if whole_experiment:
        return parse_experiment(experiment, output, parser=parser, only_new=only_new)
    return [parse_generation(folder, output, parser=parser, only_new=only_new)]
