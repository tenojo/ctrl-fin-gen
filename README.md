# ctrl-fin-gen

A Python package and CLI for controlled Finnish text generation for education
and research. The first working path uses OpenAI's Responses API and Batch API.
Google and vLLM remain placeholders. Evaluation and filtering are separate future
stages; generation keeps every returned outcome.

Requires Python 3.10 or newer. From this repository's root folder:

```bash
python -m pip install -e .
ctrl-fin-gen --help
```

The CLI uses `conditions/` and `experiments/` under the current working folder.
This initial version ships templates in the repository, so run the examples here.
`python -m ctrl_fin_gen` is an equivalent entry point.

## Prepare and preview an experiment

```bash
ctrl-fin-gen create science_pilot \
  --template third_grade_sciencebook_pilot \
  --sentences 5 5 \
  --max-word-length 10 \
  --lemmas koulu kirja taulu \
  --linguistic-controls Case=Gen Case=Par \
  --syntax-controls acl:relcl \
  --prompt "Kirjoita lyhyt kertomus koulupäivästä." \
  --preview
```

Preview prints the realized settings and exact system/user messages as JSON,
without creating files. Remove `--preview` to save the experiment. No API key is
needed for preparation. Audience defaults come from the template and can be
changed with `--audience "Kohderyhmän kuvaus"`.

The maximum word length uses the unit exposed by the selected template
(`characters` in the current pilots). The output-token budget below is a separate
model setting. Unsupported controls and values outside template bounds fail
before the experiment is created.

Use `--condition FILE.json` for reusable values and numeric linguistic targets;
CLI values override the corresponding fields. A new experiment can reuse an
existing experiment's values:

```bash
ctrl-fin-gen create science_comparison \
  --template third_grade_sciencebook_pilot \
  --copy-from science_pilot \
  --sentences 4 4
```

The system template stays under `conditions/system_templates/`. Only its
reference, realized values, prompt text/layout, and selected control definitions
are saved under the experiment's `template/` folder. Shared definitions remain
under `conditions/linguistic_controls/`.

## Generate texts

Set `OPENAI_API_KEY` in your environment before generation. Use a model available
to your account in place of `MODEL`. Model and reasoning-effort compatibility are
checked by the API; the CLI does not maintain a hard-coded model list.

```bash
ctrl-fin-gen generate science_pilot \
  --name openai_stream --backend openai --model MODEL \
  --max-length 2000 --samples 5 --stream
```

`--reasoning-effort low` is optional. Without `--stream` or `--batch`, requests run
sequentially without token streaming. If `--name` is omitted, the CLI asks for it.
Experiment and generation names must be new, single folder names.

For a batch:

```bash
ctrl-fin-gen generate science_pilot \
  --name openai_batch --backend openai --model MODEL \
  --max-length 2000 --samples 5 --batch
ctrl-fin-gen status science_pilot openai_batch
ctrl-fin-gen collect science_pilot openai_batch
```

`status` checks the saved job. `collect` downloads available output/error files
and writes normalized results, matched by sample ID. It is safe to repeat.
If a completed batch returns fewer records than requested, collection reports
`missing_results` and the remaining count without inventing replacement outputs.

## Results and recovery

Each submission lives in `experiments/EXPERIMENT/generations/NAME/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Saved model/settings, realized condition snapshot, source hashes |
| `requests.jsonl` | Exact prompts, resolved vocabulary, payloads, stable sample IDs |
| `results.jsonl` | One normalized outcome per finished sample, including response text |
| `state.json` | Progress, active request, batch job ID |
| `attempts.jsonl` | Immediate-request attempt history and errors |
| `raw/` | Full response bodies, streamed events, batch job and downloaded files |

Streaming text is displayed on stderr; command summaries are JSON on stdout.
Every terminal response is saved before the next request. Partial responses,
refusals, and provider errors are retained with their status; they are not
filtered out. Unfinished streaming deltas remain in `raw/events.jsonl`.

```bash
ctrl-fin-gen generate science_pilot --resume openai_stream
```

Resume reuses saved settings and prompts, skips finished samples, and retrieves
a known response ID where possible. It does not continue from an exact token.
Responses are submitted with `store: true` to support retrieval. The SDK's automatic
retries are disabled so attempts remain explicit.

When a connection fails before a response/job ID is saved, resume first tries to
reconcile batch submissions by their metadata. If the outcome is still unknown,
it stops. To explicitly allow another attempt, which may create a duplicate
remote request and additional API charges:

```bash
ctrl-fin-gen generate science_pilot --resume openai_stream --retry-unknown
```

After a hard process crash, a submission's `.lock` file may remain. Remove it only
once you have confirmed that no process is writing to that submission. An
unfinished final JSONL line is backed up as `.partial` before recovery.

New submissions validate source-template/catalogue hashes. If you edit those
sources, create a new experiment; existing submissions can still resume their
frozen requests.

## Vocabulary selection

`--lemmas` uses exactly the supplied list, including an explicitly empty list.
`--lemmas-file` accepts a UTF-8 text list (one lemma per line) or a JSON list.
Sampling happens during preparation:

```bash
ctrl-fin-gen create sampled_pilot --template template_pilot \
  --vocabulary-pool words.txt --sample-lemmas 3 --seed 42 \
  --selection-scope request
```

The default scope is `condition`: one selected list for repeated generations.
`request` freezes the pool and seed, then prepares each sample's list before any
API requests. Resume never redraws vocabulary. Seed `0` is valid. CSV imports
require an explicit JSON import specification; see [input examples](conditions/inputs.md).
Frequency/Zipf complexity controls require the future Parsebank integration and
currently fail clearly if selected.

## Python use and tests

The CLI calls ordinary package functions, which can also be used from research
scripts or a future GUI:

```python
from pathlib import Path
from ctrl_fin_gen.prepare import prepare_experiment
from ctrl_fin_gen.prompts import build_messages

prepared = prepare_experiment(Path.cwd(), "third_grade_sciencebook_pilot")
messages = build_messages(prepared)
```

Run the automated tests after installation:

```bash
python -m unittest discover -s tests -v
```

Tests use temporary experiment folders, fake providers, and the real OpenAI SDK
with an in-memory HTTP transport. They require no credentials or paid requests.
They cover preparation, CLI commands, streaming, interruption recovery, batch
collection, and preservation of partial/error outputs. A live API smoke test is
still needed to verify your account and chosen model.

See [conditions](conditions/README.md) for the three template tiers. Older planning
artifacts predate the runtime manifest format; create a fresh experiment and
submission for testing.
