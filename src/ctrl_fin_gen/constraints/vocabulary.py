"""Read lemma sources and prepare reproducible vocabulary choices.

read_lemmas imports lists; resolve_vocabulary applies the system template's
allowed count and stores provenance. Condition-scope sampling draws once during
experiment preparation. Request-scope sampling freezes the pool/count/seed in the
realized template; generation.py calls vocabulary_for_request for each sample
before sending anything to a provider. A sampling seed controls word selection,
not the language model's generation randomness.

These helpers normalize supplied strings but do not lemmatize them or check that
a generated text uses them. Parser-based satisfaction checks are future work.
"""

import csv
import random
import secrets
import unicodedata
from pathlib import Path

from ..storage import content_hash, read_json


def clean_lemmas(items, unique=False):
    """Return trimmed NFC-normalized strings, checking duplicates without regard to case.

    Reject nonlists, empty/nonstring items, and duplicates with ValueError.
    unique=True instead keeps the first occurrence in input order, preserving
    its spelling/case. Normalization does not infer a word's linguistic lemma.
    """
    if not isinstance(items, list):
        raise ValueError('Lemmas must be a list.')
    cleaned = []
    seen = set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('Every lemma must be a nonempty string.')
        lemma = unicodedata.normalize('NFC', item.strip())
        key = lemma.casefold()
        if key in seen:
            if unique:
                continue
            raise ValueError(f'Duplicate lemma: {lemma}')
        seen.add(key)
        cleaned.append(lemma)
    return cleaned


def read_lemmas(path):
    """Return (lemma_list, provenance) from text, JSON, or a CSV import specification.

    Text uses UTF-8 with one item per nonempty line. A JSON list is read directly;
    a JSON object must name the CSV path, encoding, delimiter, and lemma column.
    Optional import settings filter rows, remove '#' compound markers, or keep
    unique lemmas. The CSV path resolves relative to its import specification.
    Direct CSV input is rejected rather than guessing those choices.

    Provenance includes input paths/hashes and any import settings/source hash.
    Data errors raise ValueError; file/decoding errors propagate to the caller.
    """
    path = Path(path).resolve()
    provenance = {'path': str(path), 'sha256': content_hash(path)}
    if path.suffix.lower() != '.json':
        if path.suffix.lower() == '.csv':
            raise ValueError('For CSV vocabulary, supply a JSON import specification with column, encoding, and delimiter.')
        return clean_lemmas([line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]), provenance
    data = read_json(path)
    if isinstance(data, list):
        return clean_lemmas(data), provenance
    required = {'path', 'encoding', 'delimiter', 'column'}
    allowed = required | {'where', 'remove_compound_markers', 'duplicate_policy'}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - allowed:
        raise ValueError('A CSV import needs path, encoding, delimiter, and column; optional settings are where, remove_compound_markers, and duplicate_policy.')
    # An explicit import recipe makes filtering/cleanup reproducible and auditable.
    source = (path.parent / data['path']).resolve()
    items = []
    with source.open(encoding=data['encoding'], newline='') as handle:
        reader = csv.DictReader(handle, delimiter=data['delimiter'])
        columns = {data['column'], *data.get('where', {})}
        if not columns <= set(reader.fieldnames or []):
            raise ValueError('The CSV does not contain the configured columns.')
        for row in reader:
            if all(row[key] == str(value) for key, value in data.get('where', {}).items()):
                lemma = row[data['column']]
                items.append(lemma.replace('#', '') if data.get('remove_compound_markers') else lemma)
    policy = data.get('duplicate_policy', 'error')
    if policy not in ('error', 'unique'):
        raise ValueError('duplicate_policy must be error or unique.')
    provenance.update({'import': data, 'source_path': str(source), 'source_sha256': content_hash(source)})
    return clean_lemmas(items, unique=policy == 'unique'), provenance


def resolve_vocabulary(value, specification):
    """Return (realized_vocabulary, provenance) for an inline/file/sample selection.

    value contains a selection recipe; specification is the system template's
    vocabulary category, including permitted counts and minimum occurrences.
    Explicit lists and condition-scope draws produce fixed items. Request-scope
    draws instead store empty items plus a frozen sampling pool/count/seed.
    The requested count is checked even when no individual draw exists yet.

    A missing seed is generated and recorded; seed 0 remains a valid explicit
    seed. This function reads source files but writes no experiment data.
    """
    if not isinstance(value, dict) or set(value) != {'selection'}:
        raise ValueError('Vocabulary requires a selection object.')
    selection = value['selection']
    if not isinstance(selection, dict):
        raise ValueError('Vocabulary selection must be a JSON object.')
    mode = selection.get('mode')
    fields = {'inline': {'mode', 'items'}, 'file': {'mode', 'path'}, 'sample': {'mode', 'pool', 'count', 'seed', 'scope'}}
    if mode not in fields or set(selection) - fields[mode]:
        raise ValueError('Use an inline, file, or sample vocabulary selection.')
    provenance = {'mode': mode}
    sampling = None
    if mode == 'inline':
        items = clean_lemmas(selection.get('items', []))
        count = len(items)
    else:
        source_key = 'path' if mode == 'file' else 'pool'
        if not isinstance(selection.get(source_key), str):
            raise ValueError(f'Vocabulary selection needs a {source_key} path.')
        items, source = read_lemmas(selection[source_key])
        provenance.update(source)
        count = len(items)
        if mode == 'sample':
            count = selection.get('count')
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(items):
                raise ValueError('Sample count must be a nonnegative integer within the vocabulary pool size.')
            seed = selection.get('seed')
            if seed is None:
                seed = secrets.randbits(64)
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError('The sampling seed must be an integer.')
            scope = selection.get('scope', 'condition')
            if scope not in ('condition', 'request'):
                raise ValueError('Sampling scope must be condition or request.')
            provenance.update({'seed': seed, 'scope': scope, 'count': count})
            if scope == 'request':
                # Save the pool itself, so generation/resume never needs to reread
                # a vocabulary file that may later change or disappear.
                sampling = {'pool': items, 'count': count, 'seed': seed}
                items = []
            else:
                items = random.Random(seed).sample(items, count)
    limits = specification['count_range']
    if count < limits['min'] or (limits['max'] is not None and count > limits['max']):
        raise ValueError('The number of lemmas is outside the system template bounds.')
    result = {'items': items, 'min_occurrences_per_item': specification.get('min_occurrences_per_item', 1)}
    if sampling is not None:
        result['sampling'] = sampling
    return result, provenance


def vocabulary_for_request(value, sample_index):
    """Return the lemma list for a zero-based sample index in a prepared submission.

    Fixed selections return their stored items. Request-scope sampling derives a
    local random generator from the saved seed and index, making draws repeatable
    regardless of execution order. generation.py saves the resulting list into
    requests.jsonl, so actual resume reads it instead of drawing again.
    """
    if 'sampling' not in value:
        return value['items']
    sampling = value['sampling']
    # The index makes each draw independent of execution order or interruptions.
    return random.Random(f"{sampling['seed']}:{sample_index}").sample(sampling['pool'], sampling['count'])
