"""Resolve permitted experiment values without calling a generation model."""

from copy import deepcopy
from pathlib import Path

from .constraints.length import bounds, number, validate_length
from .constraints.vocabulary import clean_lemmas, resolve_vocabulary
from .models import ExperimentInputs
from .prompts import build_messages
from .storage import content_hash, named_folder, read_json, timestamp, write_json


CATALOGUE_GROUPS = (
    'linguistic_constraints',
    'syntax_tree_constraints',
    'part_of_speech_constraints',
    'vocabulary_complexity_constraints',
)


def system_template_path(root, template):
    path = Path(template)
    if not path.is_absolute():
        candidate = Path(root) / path
        if candidate.exists():
            path = candidate
        else:
            path = named_folder(Path(root) / 'conditions/system_templates', str(path))
    return (path / 'template.json' if path.is_dir() else path).resolve()


def load_values(path):
    """Paths in a condition file resolve relative to that file."""
    path = Path(path).resolve()
    values = read_json(path)
    if not isinstance(values, dict):
        raise ValueError('Condition values must be a JSON object.')
    constraints = values.get('constraints', {})
    if not isinstance(constraints, dict):
        raise ValueError('Constraints must be a JSON object.')
    vocabulary = constraints.get('vocabulary_constraints', {})
    if not isinstance(vocabulary, dict):
        raise ValueError('Vocabulary requires a selection object.')
    selection = vocabulary.get('selection', {})
    if not isinstance(selection, dict):
        raise ValueError('Vocabulary selection must be a JSON object.')
    for key in ('path', 'pool'):
        if key in selection:
            selection[key] = str((path.parent / selection[key]).resolve())
    if 'system_template' in values:
        values['system_template'] = str((path.parent / values['system_template']).resolve())
    return values


def load_catalogues(folder, system):
    catalogues = {}
    references = {}
    for category, specification in system['categories'].items():
        path = (folder / specification['catalogue_file']).resolve()
        catalogue = read_json(path)
        allowed = specification.get('allowed_controls', list(specification.get('controls', {})))
        if category == 'vocabulary_constraints':
            allowed = [specification['control']]
        if set(allowed) - set(catalogue['controls']):
            raise ValueError(f'{category} exposes keys missing from its catalogue.')
        catalogues[category] = catalogue['controls']
        references[category] = {
            'id': catalogue['catalogue_id'], 'version': catalogue['catalogue_version'],
            'path': str(path), 'sha256': content_hash(path),
        }
    return catalogues, references


def resolve_controls(items, specification, catalogue, label):
    if not isinstance(items, list):
        raise ValueError(f'{label} items must be a list.')
    limits = specification['count_range']
    if len(items) < limits['min'] or (limits['max'] is not None and len(items) > limits['max']):
        raise ValueError(f'{label} has too many or too few selected controls.')
    resolved = []
    seen = set()
    for value in items:
        item = {'control': value} if isinstance(value, str) else deepcopy(value)
        if not isinstance(item, dict) or set(item) - {'control', 'occurrences', 'target', 'proportion'}:
            raise ValueError(f'Invalid target in {label}.')
        code = item.get('control')
        if not isinstance(code, str) or code not in specification['allowed_controls'] or code in seen:
            raise ValueError(f'{label}: unexposed or duplicate control {code}.')
        seen.add(code)
        modes = catalogue[code]['target_modes']
        targets = set(item) & {'occurrences', 'target', 'proportion'}
        if len(targets) > 1:
            raise ValueError(f'{code} needs one target mode, not several.')
        if 'target' in item:
            if 'range' not in modes:
                raise ValueError(f'{code} does not accept a numeric target.')
            bounds(item['target'], code)
        elif 'proportion' in item:
            if 'token_proportion_range' not in modes:
                raise ValueError(f'{code} does not accept a proportion target.')
            bounds(item['proportion'], code, 0, 1)
        else:
            if 'presence' not in modes:
                raise ValueError(f'{code} requires a numeric target with min and max.')
            default = specification.get('default_occurrences', {'min': 1, 'max': None})
            item['occurrences'] = item.get('occurrences', deepcopy(default))
            bounds(item['occurrences'], code, minimum=0, integer=True, open_max=True)
        if label == 'vocabulary_complexity_constraints':
            raise ValueError('Vocabulary complexity controls require the future Parsebank integration.')
        resolved.append(item)
    return {'items': resolved}


def validate_realized(system, realized, catalogues):
    """Validate saved values as well as newly prepared ones."""
    constraints = realized['constraints']
    if set(constraints) != set(system['categories']):
        raise ValueError('Realized categories must match the selected system template.')
    for category, specification in system['categories'].items():
        value = constraints[category]
        if category == 'length_constraints':
            validate_length(value, specification)
        elif category == 'vocabulary_constraints':
            items = clean_lemmas(value['items'])
            sampling = value.get('sampling')
            count = len(items)
            if sampling:
                pool = clean_lemmas(sampling['pool'])
                count = number(sampling['count'], 'sample count', integer=True)
                number(sampling['seed'], 'sample seed', integer=True)
                if count < 0 or count > len(pool):
                    raise ValueError('The frozen vocabulary pool cannot supply that count.')
            limits = specification['count_range']
            if count < limits['min'] or (limits['max'] is not None and count > limits['max']):
                raise ValueError('The lemma count exceeds the system template limits.')
            if value['min_occurrences_per_item'] != specification.get('min_occurrences_per_item', 1):
                raise ValueError('Lemma occurrence requirements cannot redefine the system template.')
        else:
            resolve_controls(value['items'], specification, catalogues[category], category)
    for limit in system.get('selection_limits', {}).values():
        count = sum(len(constraints[category]['items']) for category in limit['categories'])
        if count < limit['min'] or (limit['max'] is not None and count > limit['max']):
            raise ValueError('The combined number of linguistic controls exceeds the system template limits.')
    fields = system.get('request_fields', {})
    for key in ('audience', 'prompt'):
        settings = fields.get(key, {})
        if not settings.get('tunable', False) and realized.get(key) != settings.get('default'):
            raise ValueError(f'The system template does not expose {key} for editing.')
    audience = realized.get('audience')
    if not isinstance(audience, dict) or set(audience) != {'description'}:
        raise ValueError('Audience needs a descriptive string, not a numeric grade.')
    if not isinstance(audience['description'], str):
        raise ValueError('The audience description must be text.')
    if not isinstance(realized.get('prompt'), str):
        raise ValueError('The style/theme prompt must be text.')
    maximum = system['categories'].get('length_constraints', {}).get('controls', {}).get('max_word_length')
    unit = realized.get('word_length_unit')
    if unit is not None:
        if not maximum or unit not in maximum.get('allowed_units', [maximum.get('unit')]):
            raise ValueError('That word-length unit is not exposed by this system template.')


def prepare_experiment(root, template, values=None, copy_from=None):
    path = system_template_path(root, template)
    system = read_json(path)
    catalogues, references = load_catalogues(path.parent, system)
    source = {
        'path': str(path), 'id': system['template_id'],
        'version': system['template_version'], 'sha256': content_hash(path),
    }
    fields = system.get('request_fields', {})
    selection = {
        'constraints': {},
        'audience': deepcopy(fields.get('audience', {}).get('default', {'description': ''})),
        'prompt': fields.get('prompt', {}).get('default', ''),
    }
    copied_vocabulary = None
    copied_provenance = None
    if copy_from:
        copied = load_experiment(root, copy_from)
        old_source = copied.template['source_system_template']
        if any(old_source.get(key) != source[key] for key in ('id', 'version', 'sha256')):
            raise ValueError('--copy-from must use the same system-template definition.')
        selection = deepcopy(copied.selection)
        selection['audience'] = deepcopy(copied.template['audience'])
        selection['prompt'] = copied.template['prompt']
        copied_vocabulary = deepcopy(copied.template['constraints'].get('vocabulary_constraints'))
        copied_provenance = deepcopy(copied.template['selection_provenance'].get('vocabulary_constraints'))
        for category, targets in copied.template['constraints'].items():
            if category == 'length_constraints':
                selection['constraints'][category] = deepcopy(targets)
            elif category in CATALOGUE_GROUPS:
                selection['constraints'][category] = {
                    'selection': {'mode': 'inline', 'items': deepcopy(targets['items'])},
                }

    # Merge whole targets; min/max must stay together. Length measures are independent.
    values = deepcopy(values if values is not None else {})
    allowed = {
        'constraints', 'audience', 'prompt', 'prompts', 'word_length_unit', 'metadata',
        'schema_version', 'condition_id', 'artifact_type', 'system_template',
    }
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError('Condition values must contain only recognized fields.')
    if values.get('system_template') and Path(values['system_template']).resolve() != path:
        raise ValueError('The condition file names a different system template.')
    for key, value in values.items():
        if key == 'constraints':
            if not isinstance(value, dict):
                raise ValueError('Constraints must be a JSON object.')
            for category, targets in value.items():
                if category == 'length_constraints' and isinstance(targets, dict):
                    selection['constraints'].setdefault(category, {}).update(targets)
                else:
                    selection['constraints'][category] = targets
        elif key == 'prompts':
            if not isinstance(value, dict):
                raise ValueError('Prompt overrides must be a JSON object.')
            selection.setdefault('prompts', {}).update(value)
        else:
            selection[key] = value
    if set(selection['constraints']) - set(system['categories']):
        raise ValueError('The request adds a category not exposed by this system template.')

    prompts = {}
    for key in ('system', 'user_prefix', 'user_reminder', 'layout'):
        filename = system['prompts'].get(key + '_file')
        prompts[key] = (path.parent / filename).read_text(encoding='utf-8') if filename else ''
    overrides = selection.get('prompts', {})
    if set(overrides) - {'system', 'user_prefix', 'user_reminder'}:
        raise ValueError('Unknown prompt override; the layout belongs to the system template.')
    for key, text in overrides.items():
        if key + '_file' not in system['prompts'] or not isinstance(text, str):
            raise ValueError(f'The system template does not expose prompt text {key}.')
        prompts[key] = text

    realized = {
        'artifact_type': 'realized_template', 'schema_version': system.get('schema_version'),
        'source_system_template': source, 'catalogues': references, 'constraints': {},
        'audience': selection['audience'], 'prompt': selection['prompt'],
        'prompts': deepcopy(system['prompts']), 'selection_provenance': {},
        'metadata': deepcopy(selection.get('metadata', {})),
    }
    for category, specification in system['categories'].items():
        supplied = selection['constraints'].get(category)
        if category == 'vocabulary_constraints':
            if copied_vocabulary is not None and category not in values.get('constraints', {}):
                realized['constraints'][category] = copied_vocabulary
                realized['selection_provenance'][category] = copied_provenance
            else:
                if supplied is None:
                    supplied = {'selection': {'mode': 'inline', 'items': specification['default_items']}}
                value, provenance = resolve_vocabulary(supplied, specification)
                realized['constraints'][category] = value
                realized['selection_provenance'][category] = provenance
        elif category == 'length_constraints':
            realized['constraints'][category] = validate_length({} if supplied is None else supplied, specification)
        else:
            if supplied is None:
                items = specification['default_items']
            else:
                if not isinstance(supplied, dict) or set(supplied) != {'selection'}:
                    raise ValueError(f'{category} requires an inline selection of control targets.')
                choice = supplied['selection']
                if not isinstance(choice, dict) or set(choice) != {'mode', 'items'} or choice['mode'] != 'inline':
                    raise ValueError(f'{category} requires mode=inline and an items list.')
                items = choice['items']
            realized['constraints'][category] = resolve_controls(items, specification, catalogues[category], category)
    maximum = system['categories'].get('length_constraints', {}).get('controls', {}).get('max_word_length')
    realized['word_length_unit'] = selection.get('word_length_unit', maximum.get('unit') if maximum else None)
    validate_realized(system, realized, catalogues)
    prepared = ExperimentInputs(selection, realized, catalogues, prompts)
    build_messages(prepared)  # Validate layout before creating any files.
    return prepared


def create_experiment(root, name, template, values=None, copy_from=None, preview=False):
    destination = named_folder(Path(root) / 'experiments', name)
    if destination.exists() and not preview:
        raise FileExistsError(f'Experiment already exists: {destination}')
    prepared = prepare_experiment(root, template, values, copy_from)
    if preview:
        return prepared
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # No overwrite, including if another process wins the race.
    folder = destination / 'template'
    folder.mkdir()
    prepared.template.update({'experiment': name, 'created_at': timestamp()})
    for key in prepared.prompts:
        filename = 'prompt_layout.txt' if key == 'layout' else key + '.txt'
        prepared.template['prompts'][key + '_file'] = filename
    write_json(folder / 'selection.json', prepared.selection)
    write_json(folder / 'template.json', prepared.template)
    used = {}
    for category, value in prepared.template['constraints'].items():
        if category == 'vocabulary_constraints':
            keys = ['required_lemmas'] if value['items'] or value.get('sampling') else []
        elif category == 'length_constraints':
            keys = [key for key, target in value.items() if target is not None]
        else:
            keys = [item['control'] for item in value['items']]
        used[category] = {key: prepared.controls[category][key] for key in keys}
    write_json(folder / 'control_definitions.json', used)
    for key, text in prepared.prompts.items():
        (folder / prepared.template['prompts'][key + '_file']).write_text(text, encoding='utf-8')
    return prepared


def load_experiment(root, name):
    folder = named_folder(Path(root) / 'experiments', name) / 'template'
    realized = read_json(folder / 'template.json')
    source = realized['source_system_template']
    path = (folder / source['path']).resolve()
    if content_hash(path) != source['sha256']:
        raise ValueError('The source system template has changed; create an experiment from its new version.')
    system = read_json(path)
    controls, references = load_catalogues(path.parent, system)
    for category, reference in realized['catalogues'].items():
        if reference['sha256'] != references[category]['sha256']:
            raise ValueError('A shared control catalogue has changed; create a new experiment.')
    validate_realized(system, realized, controls)
    prompts = {}
    for key in ('system', 'user_prefix', 'user_reminder', 'layout'):
        path = folder / realized['prompts'][key + '_file']
        prompts[key] = path.read_text(encoding='utf-8')
    selection = read_json(folder / 'selection.json')
    return ExperimentInputs(selection, realized, controls, prompts)
