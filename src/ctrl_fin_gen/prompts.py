"""Render prepared constraints as Finnish instructions and system/user messages.

build_messages consumes models.ExperimentInputs from prepare.py. It first renders
each constraint category, groups categories into layout sections, then fills the
system template's prompt_layout.txt. The pilot order is prefix, length,
vocabulary, morphosyntax, audience, theme/style, and reminder; the layout file
controls that order rather than a fixed prompt string in Python.

Shared Finnish wording comes from conditions/linguistic_controls/. To change
phrasing, edit those definitions or the template's prompt text/layout; to support
a new target representation, extend control_instruction and preparation validation.
No files are written and no provider is called here. A provider can translate the
returned role/content messages into its own API request format.
"""

from string import Formatter


def control_instruction(code, target, definition):
    """Return one Finnish instruction for a validated feature target.

    code identifies the feature, target contains its resolved occurrence/range/
    proportion values, and definition supplies catalogue wording. Presence uses
    the catalogue's complete instruction (including the phrase after 'käyttää');
    nondefault occurrence counts and proportions use the formats below.
    """
    if 'target' in target:
        return definition['range_instruction_template_fi'].format(**target['target'])
    if 'proportion' in target:
        limits = target['proportion']
        return f"Piirteen {code} esiintymien määrän suhteen tekstin tokenimäärään tulee olla välillä {limits['min']}–{limits['max']}."
    counts = target['occurrences']
    if counts == {'min': 1, 'max': None}:
        return definition['presence_instruction_fi']
    phrase = definition['prompt_description_fi']
    high = counts['max']
    count_text = f"vähintään {counts['min']}" if high is None else f"{counts['min']}–{high}"
    return f'Tekstissä tulee käyttää seuraavaa piirrettä {count_text} kertaa: {phrase} ({code}).'


def build_messages(experiment, vocabulary=None):
    """Return a list of role/content dictionaries for one prepared text request.

    experiment is validated ExperimentInputs. vocabulary=None uses its realized
    lemma list; a supplied list (including []) overrides it for one request.
    generation.start_generation supplies each frozen draw for request-scope
    sampling. A preparation preview has no draw yet and therefore omits that
    request's vocabulary instruction.

    Disabled/empty groups are omitted. Only known, plain layout placeholders are
    accepted; unsupported placeholders raise ValueError. A nonempty system prompt
    becomes a separate message, followed by one assembled user message. The
    returned structure is independent of an SDK object, but currently placed in
    an OpenAI Responses payload by generation.py.
    """
    values = experiment.template
    groups = {}
    for category, value in values['constraints'].items():
        lines = []
        catalogue = experiment.controls[category]
        if category == 'vocabulary_constraints':
            items = value['items'] if vocabulary is None else vocabulary
            if items:
                text = catalogue['required_lemmas']['instruction_template_fi'].format(lemmas=', '.join(items))
                lines.append(text)
        elif category == 'length_constraints':
            for key, target in value.items():
                if target is None:
                    continue
                definition = catalogue[key]
                if key == 'max_word_length':
                    text = definition['instruction_templates_fi'][values['word_length_unit']].format(value=target)
                elif target['min'] == target['max']:
                    text = definition['exact_instruction_template_fi'].format(value=target['min'])
                else:
                    text = definition['range_instruction_template_fi'].format(**target)
                lines.append(text)
        else:
            lines = [control_instruction(item['control'], item, catalogue[item['control']]) for item in value['items']]
        groups[category] = '\n'.join(lines)
    # A layout section can combine several catalogue categories, for example
    # morphology + syntax + POS under the morphosyntactic section.
    sections = {
        'user_prompt_prefix': experiment.prompts['user_prefix'].strip(),
        'user_prompt_reminder': experiment.prompts['user_reminder'].strip(),
        'audience_constraint': 'Kohderyhmä:\n' + values['audience']['description'].strip() if values['audience']['description'].strip() else '',
        'style_or_thematic_constraint': values['prompt'].strip(),
    }
    defaults = {'length_constraints': ['length_constraints'], 'vocabulary_constraints': ['vocabulary_constraints'], 'morphosyntactic_constraints': ['linguistic_constraints', 'syntax_tree_constraints', 'part_of_speech_constraints']}
    for section, categories in {**defaults, **values['prompts'].get('section_groups', {})}.items():
        sections[section] = '\n'.join(groups.get(category, '') for category in categories if groups.get(category))
    # Work paragraph by paragraph: omit a paragraph whose placeholders are all
    # empty, and reject attribute access/conversions/format specs in placeholders.
    blocks = []
    for block in experiment.prompts['layout'].strip().split('\n\n'):
        fields = []
        for _, field, specification, conversion in Formatter().parse(block):
            if field is not None:
                if field not in sections or specification or conversion:
                    raise ValueError(f'Unknown or unsupported prompt-layout placeholder: {field}')
                fields.append(field)
        if fields and not any(sections[field] for field in fields):
            continue
        text = block.format(**sections).strip()
        if text:
            blocks.append(text)
    messages = []
    if experiment.prompts['system'].strip():
        messages.append({'role': 'system', 'content': experiment.prompts['system'].strip()})
    messages.append({'role': 'user', 'content': '\n\n'.join(blocks)})
    return messages


def build_prompt(experiment, vocabulary=None):
    """Return backend-neutral system/user text. Message-list rendering is kept for preview compatibility."""
    messages = build_messages(experiment, vocabulary)
    return {
        'system_prompt': next((m['content'] for m in messages if m['role'] == 'system'), ''),
        'user_prompt': next(m['content'] for m in messages if m['role'] == 'user'),
    }
