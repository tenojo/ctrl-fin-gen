"""Extract strict lemma/UD counts and character-based lengths from CoNLL-U."""

from collections import Counter
from pathlib import Path

from ._files import digest, read_json, write_json

EXTRACTOR_VERSION = '2'
POLICIES = {
    'matching': 'exact_case_sensitive', 'lemmas': 'remove_#_only',
    'words': 'whitespace_separated_FORMS_excluding_PUNCT_and_SYM',
    'word_length': 'Unicode_characters', 'proportion_denominator': 'all_integer_ID_rows',
}


def extract_features(conllu: str):
    """Measure a document; ignore MWT headers/empty nodes and retain exact tags.

    Sentence boundaries are blank lines. FEATS '_' means no morphological tags,
    whereas absent lemmas/POS/deprels make their respective checks unavailable.
    """
    counts = {group: Counter() for group in (
        'vocabulary_constraints', 'linguistic_constraints',
        'syntax_tree_constraints', 'part_of_speech_constraints',
    )}
    missing, unreliable = Counter(), Counter()
    words, token_count, sentences, in_sentence = [], 0, 0, False
    for line in conllu.splitlines() + ['']:
        if not line.strip():
            sentences += int(in_sentence)
            in_sentence = False
            continue
        if line.startswith('#'):
            continue
        columns = line.split('\t')
        if len(columns) != 10:
            raise ValueError('Expected ten tab-separated CoNLL-U columns.')
        token_id, form, lemma, upos, _, feats, head, deprel, _, _ = columns
        if '-' in token_id or '.' in token_id:
            continue
        if not token_id.isdecimal() or int(token_id) < 1 or not form or form == '_':
            raise ValueError('Invalid CoNLL-U word ID or FORM.')
        in_sentence, token_count = True, token_count + 1
        if len(form.split()) > 1:
            # Imported/old CoNLL-U may still contain a merged word row. Its
            # orthographic lengths are measurable; its linguistic labels aren't
            # reliable measurements for the component words. Never guess lemmas.
            unreliable.update(counts.keys())
        for group, value in (
            ('vocabulary_constraints', lemma.replace('#', '')),
            ('part_of_speech_constraints', upos), ('syntax_tree_constraints', deprel),
        ):
            if value and value != '_':
                counts[group][value] += 1
            else:
                missing[group] += 1
        if feats != '_':
            for feature in feats.split('|'):
                if '=' not in feature:
                    raise ValueError(f'Invalid morphological feature: {feature}')
                counts['linguistic_constraints'][feature] += 1
        if upos not in ('PUNCT', 'SYM'):
            words.extend(form.split())
        if head == '0' and deprel == 'root' and upos in ('NOUN', 'VERB', 'ADJ'):
            counts['part_of_speech_constraints'][upos + '_headed_phrase'] += 1
    if not token_count:
        raise ValueError('CoNLL-U contains no word rows.')
    return {
        **{group: dict(value) for group, value in counts.items()},
        'length_constraints': {'number_of_words': len(words), 'number_of_sentences': sentences,
                               'max_word_length': max(map(len, words), default=0)},
        'word_lengths': {word: len(word) for word in words},
        'token_count': token_count, 'missing_annotations': dict(missing),
        'unreliable_annotations': dict(unreliable),
    }


def extract_folder(input_folder, output_folder=None, *, only_new=False):
    """Write one feature JSON per .conllu, defaulting to sibling feature_analysis/."""
    folder = Path(input_folder).resolve()
    if not folder.is_dir():
        raise ValueError(f'No CoNLL-U folder: {folder}')
    output = Path(output_folder).resolve() if output_folder else folder.parent / 'feature_analysis'
    if output == folder:
        raise ValueError('Feature output must be separate from the CoNLL-U folder.')
    output.mkdir(parents=True, exist_ok=True)
    counts = {'output_folder': str(output), 'extracted': 0, 'reused': 0, 'errors': 0}
    for path in sorted(folder.glob('*.conllu')):
        text = path.read_text(encoding='utf-8')
        fingerprint = digest(text)
        target = output / (path.stem + '.json')
        old = read_json(target) if only_new and target.exists() else {}
        if (only_new and old.get('status') == 'extracted' and old.get('conllu_sha256') == fingerprint
                and old.get('extractor_version') == EXTRACTOR_VERSION):
            counts['reused'] += 1
            continue
        result = {'text_id': path.stem, 'conllu_sha256': fingerprint,
                  'extractor_version': EXTRACTOR_VERSION, 'policies': POLICIES}
        try:
            result.update(status='extracted', features=extract_features(text))
            counts['extracted'] += 1
        except ValueError as error:
            result.update(status='error', error=str(error))
            counts['errors'] += 1
        write_json(target, result)
    return counts
