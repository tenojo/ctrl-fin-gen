"""Pure category comparisons, plus writing one short report per saved text."""

import json
from pathlib import Path
import re

from ._files import digest, read_json, read_jsonl, write_text
from .feature_extractor import EXTRACTOR_VERSION
from .parser_process import PARSING_VERSION

UPOS = set('ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X'.split())
CHECKER_VERSION = '2'


def _compare(group, feature, target, realized, reason=None):
    status = 'unavailable' if reason else 'failed'
    if not reason:
        if not isinstance(target, dict) or not set(target) <= {'min', 'max'}:
            status, reason = 'unsupported', 'Unsupported target.'
        else:
            low, high = target.get('min'), target.get('max')
            if realized is None:
                status, reason = 'unavailable', 'No measurement is available.'
            elif ((low is None or realized >= low) and (high is None or realized <= high)):
                status = 'passed'
    return {'group': group, 'feature': feature, 'target': target,
            'realized_value': realized, 'status': status, 'reason': reason}


def check_vocabulary(features, constraints):
    group = 'vocabulary_constraints'
    counts = features.get(group, {}) if features else {}
    reason = 'Lemma annotations are unavailable.' if not features or features.get('missing_annotations', {}).get(group) else None
    if features and features.get('unreliable_annotations', {}).get(group):
        reason = 'Whitespace-spanning word tokens require re-parsing before lemma checks.'
    target = {'min': constraints.get('min_occurrences_per_item', 1), 'max': None}
    return [_compare(group, lemma, target, counts.get(lemma.replace('#', ''), 0) if not reason else None, reason)
            for lemma in constraints.get('items', [])]


def check_length(features, constraints):
    group, results = 'length_constraints', []
    for feature, target in constraints.items():
        if target is None:
            continue
        reason = 'Length measurements are unavailable.' if not features else None
        if (features and feature != 'number_of_sentences'
                and features.get('missing_annotations', {}).get('part_of_speech_constraints')):
            reason = 'Missing POS annotations prevent word counting.'
        realized = features.get(group, {}).get(feature) if features and not reason else None
        bounds = {'min': None, 'max': target} if feature == 'max_word_length' else target
        result = _compare(group, feature, bounds, realized, reason)
        if feature == 'max_word_length' and result['status'] == 'failed':
            result['words'] = [{'word': word, 'characters': length}
                               for word, length in features.get('word_lengths', {}).items() if length > target]
        if feature not in ('number_of_words', 'number_of_sentences', 'max_word_length'):
            result.update(status='unsupported', reason='Unsupported length feature.')
        results.append(result)
    return results


def _check_tags(features, constraints, group):
    results = []
    for item in constraints.get('items', []):
        item = {'control': item} if isinstance(item, str) else item
        feature = item['control']
        target = item.get('occurrences', item.get('proportion', item.get('target', {'min': 1, 'max': None})))
        supported = (
            (group == 'linguistic_constraints' and '=' in feature)
            or (group == 'syntax_tree_constraints' and re.fullmatch('[a-z]+(?::[a-z]+)?', feature))
            or (group == 'part_of_speech_constraints' and
                feature in UPOS | {'NOUN_headed_phrase', 'VERB_headed_phrase', 'ADJ_headed_phrase'})
        )
        reason = 'Annotations are unavailable.' if not features or features.get('missing_annotations', {}).get(group) else None
        if features and features.get('unreliable_annotations', {}).get(group):
            reason = 'Whitespace-spanning word tokens require re-parsing before linguistic checks.'
        realized = features.get(group, {}).get(feature, 0) if features and not reason else None
        if 'proportion' in item and not reason:
            denominator = features['token_count']
            if not denominator:
                reason, realized = 'Proportion denominator is zero.', None
            else:
                realized /= denominator
        result = _compare(group, feature, target, realized, reason)
        if not supported:
            result.update(status='unsupported', realized_value=None,
                          reason='This feature is not implemented in the first evaluator version.')
        results.append(result)
    return results


def check_linguistic(features, constraints):
    return _check_tags(features, constraints, 'linguistic_constraints')


def check_syntax(features, constraints):
    return _check_tags(features, constraints, 'syntax_tree_constraints')


def check_pos(features, constraints):
    return _check_tags(features, constraints, 'part_of_speech_constraints')


def check_constraints(features, conditions):
    """Return individual checks; conditions may be a full request or its constraints."""
    groups = conditions.get('constraints', conditions)
    checkers = {'vocabulary_constraints': check_vocabulary, 'length_constraints': check_length,
                'linguistic_constraints': check_linguistic, 'syntax_tree_constraints': check_syntax,
                'part_of_speech_constraints': check_pos}
    checks = []
    for group, targets in groups.items():
        if group in checkers:
            checks.extend(checkers[group](features, targets))
        else:
            for item in targets.get('items', []):
                feature = item if isinstance(item, str) else item['control']
                result = _compare(group, feature, item, None)
                result.update(status='unsupported', reason='Unsupported constraint group.')
                checks.append(result)
    return checks


def satisfaction_report(checks):
    failed = [c for c in checks if c['status'] != 'passed']
    failed_groups = list(dict.fromkeys(c['group'] for c in failed))
    return {
        'success': int(not failed),
        'satisfied_groups': list(dict.fromkeys(c['group'] for c in checks if c['group'] not in failed_groups)),
        'satisfied_features': list(dict.fromkeys(c['feature'] for c in checks if c['status'] == 'passed')),
        'failed_groups': failed_groups,
        'failed_features': list(dict.fromkeys(c['feature'] for c in failed)),
        'failure_reasons': [
            {key: value for key, value in c.items() if key in ('feature', 'group', 'target', 'realized_value', 'words')}
            | ({'reason': c['reason'], 'status': c['status']} if c['reason'] else {}) for c in failed
        ],
    }


def check_folder(input_folder, *, constraints_path=None, features_folder=None, output_file=None, only_new=False):
    """Check one Y folder; missing/error/stale analyses remain explicit failures."""
    folder = Path(input_folder).resolve()
    constraints_path = Path(constraints_path) if constraints_path else folder / 'constraints.json'
    features_folder = Path(features_folder) if features_folder else folder / 'feature_analysis'
    output_file = Path(output_file) if output_file else folder / 'constraint_satisfaction.jsonl'
    entries = read_json(constraints_path)['texts']
    previous = {r['text_id']: r for r in read_jsonl(output_file)} if only_new and output_file.exists() else {}
    reports, checked, reused = [], 0, 0
    for entry in entries:
        text_id = entry['text_id']
        if not re.fullmatch('[0-9a-f]{24}', text_id):
            raise ValueError(f'Invalid text_id in constraints.json: {text_id}')
        feature_path = features_folder / (text_id + '.json')
        parse_path = folder / 'conllus' / (text_id + '.json')
        conllu_path = parse_path.with_suffix('.conllu')
        feature_text = feature_path.read_text(encoding='utf-8') if feature_path.exists() else ''
        parse_text = parse_path.read_text(encoding='utf-8') if parse_path.exists() else ''
        conllu_hash = digest(conllu_path.read_text(encoding='utf-8')) if conllu_path.exists() else ''
        fingerprint = digest(CHECKER_VERSION + json.dumps(entry, sort_keys=True) + feature_text + parse_text + conllu_hash)
        if only_new and previous.get(text_id, {}).get('input_sha256') == fingerprint:
            reports.append(previous[text_id])
            reused += 1
            continue
        features, error = None, None
        try:
            parsed = json.loads(parse_text) if parse_text else {}
            extracted = json.loads(feature_text) if feature_text else {}
            if parsed.get('status') != 'parsed':
                raise ValueError(parsed.get('error', 'Parsed text is missing.'))
            if parsed.get('parsing_version') != PARSING_VERSION:
                raise ValueError('Parser processing has changed; rerun parse and extract.')
            if (parsed.get('text_sha256') != entry['text_sha256'] or digest(entry['text']) != entry['text_sha256']
                    or parsed.get('conllu_sha256') != conllu_hash):
                raise ValueError('Parsed text has changed; rerun parse and extract.')
            if extracted.get('status') != 'extracted':
                raise ValueError(extracted.get('error', 'Feature analysis is missing; run extract.'))
            if extracted.get('conllu_sha256') != conllu_hash or extracted.get('extractor_version') != EXTRACTOR_VERSION:
                raise ValueError('Feature analysis is stale; rerun extract.')
            features = extracted['features']
        except (ValueError, KeyError) as exception:
            error = str(exception)
        report = {key: entry.get(key) for key in
                  ('text_id', 'request_id', 'attempt_id', 'configuration_id', 'repetition_index')}
        report.update(satisfaction_report(check_constraints(features, entry['constraints'])))
        if error:
            report.update(success=0, error=error)
            if not report['failure_reasons']:
                report['failure_reasons'].append({'feature': 'analysis', 'target': 'available',
                                                  'realized_value': None, 'reason': error})
        report['input_sha256'] = fingerprint
        report['checker_version'] = CHECKER_VERSION
        reports.append(report)
        checked += 1
    write_text(output_file, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in reports))
    return {'output_file': str(output_file), 'checked': checked, 'reused': reused,
            'successes': sum(r['success'] for r in reports), 'errors': sum('error' in r for r in reports)}
