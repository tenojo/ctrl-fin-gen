"""Turn a system template and user choices into an experiment's realized inputs.

The three data tiers meet here: shared catalogues define controls and wording;
a system template exposes a subset with defaults/bounds; a realized template
stores the actual choices for an experiment. A request cannot add controls that
its system template does not expose.

prepare_experiment builds an ExperimentInputs object in memory. create_experiment
optionally saves it under experiments/NAME/template/. load_experiment reloads it
and checks its source hashes before a new generation starts. Precedence is system
defaults, copied experiment values, then supplied values (already merged with
explicit CLI overrides by cli.creation_values).

Vocabulary import/selection and numeric checks live in constraints/. Finnish
prompt rendering lives in prompts.py. No generation backend is called here;
this preparation path is shared by future OpenAI, Google, and vLLM clients.
"""

from copy import deepcopy
from pathlib import Path

from .constraints.length import bounds, number, validate_length
from .constraints.vocabulary import clean_lemmas, resolve_vocabulary, vocabulary_for_request
from .models import ExperimentInputs, GenerationRequest
from .prompts import build_messages, build_prompt
from .storage import content_hash, named_folder, read_json, timestamp, write_json


# These categories share the same inline control-target representation. Vocabulary
# and length have their own formats and are handled separately below.
CATALOGUE_GROUPS = (
    'linguistic_constraints',
    'syntax_tree_constraints',
    'part_of_speech_constraints',
    'vocabulary_complexity_constraints',
)


def system_template_path(root, template):
    """Return the absolute template.json path for a name, folder, or JSON path.

    Existing relative paths resolve from root; a bare name falls back to
    root/conditions/system_templates/NAME. This resolves a path but does not
    read or validate the template's JSON contents.
    """
    path = Path(template)
    if not path.is_absolute():
        candidate = Path(root) / path
        if candidate.exists():
            path = candidate
        else:
            path = named_folder(Path(root) / 'conditions/system_templates', str(path))
    return (path / 'template.json' if path.is_dir() else path).resolve()


def load_values(path):
    """Read a condition selection JSON object and make its input paths absolute.

    Vocabulary source paths and a system_template reference resolve relative to
    the condition file, rather than the shell's working directory. Return the
    choices without filling defaults or selecting vocabulary; those steps belong
    to prepare_experiment. Malformed selection structure raises ValueError.
    """
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
    """Return (catalogues, references) for the selected system template.

    folder contains its template.json. Catalogue-file paths resolve from there.
    catalogues maps each category to its shared control definitions; references
    records each file's identity, path, and hash for later consistency checks.
    Reject a template that exposes control codes absent from those definitions.
    """
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
    """Return {'items': [...]} with validated targets and presence defaults.

    Used for morphology, syntax, POS, and the reserved complexity category.
    items may contain a code string or a dictionary such as
    {'control': 'Case=Gen', 'occurrences': {'min': 1, 'max': None}}.
    A numeric feature uses target; a token proportion uses proportion. Each
    item must select one supported mode. Category size, allowed codes, duplicate
    codes, and numeric bounds are checked against the template/catalogue.

    Missing occurrence bounds mean presence (normally at least once). This
    validates requested conditions, not their satisfaction in generated text.
    Vocabulary-complexity targets remain disabled pending resource integration.
    """
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
    """Check a realized template against its source; return None or raise ValueError.

    Unlike prepare_experiment, this does not resolve choices or fill defaults.
    It checks already-realized categories, vocabulary/pool size, feature targets,
    combined selection limits, audience/theme editability, and word-length units.
    load_experiment uses the same checks as creation so saved values cannot widen
    a system template's exposed controls or bounds.
    """
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
    # Several individually valid groups can still exceed a shared constraint cap.
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
    """Return validated ExperimentInputs without writing an experiment or calling an API.

    root contains conditions/ and experiments/; template accepts a name or path.
    values is a condition selection dictionary, not a saved realized template.
    copy_from optionally names an experiment with the same source definition;
    explicit values override its choices. Inputs are copied before modification.

    The result contains defaults, resolved targets, frozen vocabulary or sampling
    pool, prompt text/layout, and source references. Prompt rendering is checked
    before returning. Invalid choices raise ValueError; file errors propagate.
    """
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
        # Copy realized values, not outputs. Reuse the frozen vocabulary/pool so
        # copying still works after an original input vocabulary file is removed.
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

    # Layout belongs to the system template. Only its exposed prompt text can
    # be overridden for this experiment; the layout's seven-section order stays.
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

    # Keep selection recipes separate from the resolved data used for generation.
    # References identify source files; the system-template definition is not copied.
    realized = {
        'artifact_type': 'realized_template', 'schema_version': system.get('schema_version'),
        'source_system_template': source, 'catalogues': references, 'constraints': {},
        'audience': selection['audience'], 'prompt': selection['prompt'],
        'prompts': deepcopy(system['prompts']), 'selection_provenance': {},
        'metadata': deepcopy(selection.get('metadata', {})),
    }
    # Each category resolves to its own saved shape: lemma items/pool, length
    # bounds, or a list of feature targets. Omitted choices use template defaults.
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
    """Prepare and return an experiment, saving it unless preview=True.

    A normal call reserves a new experiments/NAME folder and writes selection.json,
    realized template.json, selected control_definitions.json, and prompt text/
    layout under template/. Existing experiment folders are never overwritten.
    The source system-template JSON stays in conditions/system_templates/.

    Preview returns the same prepared inputs without creating any folders/files.
    All value and layout validation happens before a destination is created.
    """
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
    # These selected-definition snapshots support later auditing/evaluation.
    # The returned ExperimentInputs.controls still contains the full catalogues.
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
    """Return saved ExperimentInputs after checking source hashes and allowed values.

    Read realized values and prompt snapshots from experiments/NAME/template/.
    Reopen the referenced system template and shared catalogues; reject changed
    definitions instead of silently changing what the experiment means. The local
    selected-control snapshots are audit artifacts, not the source loaded here.

    New submissions use this path. Resuming an existing submission instead uses
    its frozen manifest/requests, so it does not need to rerender an experiment.
    """
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


def prepare_requests(experiment, samples):
    """Freeze neutral requests, including vocabulary and grouping, before provider selection.

    Condition-scope choices share a configuration with consecutive repetitions.
    Request-scope draws get distinct configuration IDs even if two draws coincide.
    Reusing this returned plan preserves identifiers and every realized choice.
    """
    vocabulary = experiment.template['constraints'].get('vocabulary_constraints', {'items': []})
    requests = []
    for index in range(samples):
        lemmas = vocabulary_for_request(vocabulary, index)
        conditions = deepcopy(experiment.template)
        conditions.pop('catalogues', None)
        conditions.pop('prompts', None)
        if 'vocabulary_constraints' in conditions['constraints']:
            realized = conditions['constraints']['vocabulary_constraints']
            realized['items'] = lemmas
            realized.pop('sampling', None)
        per_request = bool(vocabulary.get('sampling'))
        requests.append(GenerationRequest(
            request_id=f'sample_{index + 1:06d}',
            configuration_id=f'config_{index + 1:06d}' if per_request else 'config_000001',
            repetition_index=1 if per_request else index + 1,
            conditions=conditions,
            **build_prompt(experiment, lemmas),
        ))
    return requests
