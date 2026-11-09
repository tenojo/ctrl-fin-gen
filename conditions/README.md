# Controls, system templates, and realized templates

The project has three tiers, with separate responsibilities:

```text
conditions/
    linguistic_controls/             # shared definitions used by generation and evaluation
        linguistic_controls.json
        syntax_tree_controls.json
        part_of_speech_controls.json
        vocabulary_complexity_controls.json
        length_controls.json
        vocabulary_controls.json
    system_templates/
        template_pilot/
            template.json            # exposed controls, defaults, bounds, and fields
            prompt_layout.txt
            system.txt
            user_prefix.txt
            user_reminder.txt
experiments/
    <experiment>/
        template/
            template.json            # realized choices and values
```

## 1. Shared control definitions

`linguistic_controls/` is the broad catalogue. It describes feature keys, Finnish
prompt phrases, and the definitions needed by independent evaluation methods.
Control meaning belongs here, not in each system template. The same feature can
be used by many system templates without copying its definition into each one.
Length and lemma-presence checks have shared definitions too. Frequency/Zipf-based
vocabulary complexity remains a future resource integration. The definitions do
not mean that parser-backed checkers are implemented or infallible.

## 2. System templates

A system template selects a subset of the shared controls and specifies what can
be tuned: allowed keys, counts, value bounds, defaults, and request fields. It
also owns its prompt layout and system/user prompt text. Catalogue references
resolve relative to its `template.json`; prompt-file references resolve in the
same folder. The pilot uses `../../linguistic_controls/<catalogue>.json`.

For example, a template could expose only `Case=Gen` and `Tense=Past`, allow zero
to two of them to be selected, and permit a sentence count from three to seven.
A realization could then select either, both, or neither and choose five
sentences. It could not introduce `Case=Par` or request ten sentences.

The pilot's broad allowed lists are an illustrative selection, not a requirement
that every system template expose every catalogue feature. `length_constraints`
references the shared length catalogue and contains a `controls` mapping of the
exposed measures and their bounds. Vocabulary selects the shared `required_lemmas`
control and specifies permitted list sizes and defaults.

Users can create system templates by selecting available controls and setting
ranges. User-authored system templates also live under `system_templates/`.
Create a new named folder when making a derivative, leaving the original intact.
There is no separate custom-template tier or inheritance engine.

## 3. Realized templates

Actual selections, numeric values, vocabulary items, target audience, and theme
belong under `experiments/<experiment>/template/`. The realized `template.json`
records which source system template it uses. The
[static pilot example](../experiments/pilot_example/README.md) also includes the
input selection, a source-system-template reference, prompt snapshots, and
selected control-definition snapshots. The system-template definition stays under
`conditions/system_templates/`; the realized record identifies it by path, ID,
version, and content hash.

Before generation, show the selected system template's exposed fields. Users may
change those values and select only its allowed controls. Creating a new category,
adding an unexposed control, widening a bound, or changing a control definition
belongs to system-template authoring, not realization. CLI overrides obey the same
rules. Unknown fields and disallowed values must fail clearly, not be ignored.
Realized inputs are frozen once used for a run; changed inputs form a new
realization rather than overwriting the record behind existing outputs.

## Target audience

The prompt layout retains seven sections, with target audience replacing numeric
grade in the same position. `audience.description` is free-form text, for example:

- Peruskoulun 3. luokan oppilaat
- Sosiaali- ja terveysalan ammatillisen koulutuksen opiskelijat
- Lukion ensimmäisen vuoden opiskelijat

A numeric grade, education level, discipline, or study stage can be added as
optional structured metadata later. No universal grade scale is required. Audience
suitability is not treated as a deterministic grammar check.

## Arguments and implementation stage

```text
create science_pilot --template template_pilot
create science_copy --template template_pilot --copy-from science_pilot
create nursing_pilot --template template_pilot --audience "Sosiaali- ja terveysalan opiskelijat"
generate science_pilot --name model_a --backend openai --model MODEL --max-length 2000 --samples 20 --stream
status science_pilot model_a
collect science_pilot model_a
```

These subcommands follow the `ctrl-fin-gen` command name. `create` selects the
system template and writes the experiment's realized values. Optional
`--copy-from` reuses values only from an experiment with the same source system
template. Explicit overrides must remain within its exposed controls and bounds.
`--condition` can supply values as JSON. The system-template definition stays in
`conditions/system_templates/`.

`generate` reads an existing experiment's conditions and stores a named submission
under its `generations/` folder. Resumption uses `generate EXPERIMENT --resume NAME`
with saved settings. See the [complete workflow](../docs/cli.md).

Creation, validation, prompt assembly, OpenAI generation, incremental saving,
resumption, status, and collection are implemented. Values are supplied through
CLI arguments or condition JSON; `create --preview` shows the realized values
and prompts. Interactive condition editing and evaluation remain future steps.
The static example artifacts predate the working commands.
