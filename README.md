# ctrl-fin-gen

A Python package and CLI for controlled Finnish text generation for education
and research. OpenAI and the Gemini API support generation, streaming, and
background batch jobs. The vLLM server adapter shares the Responses implementation;
the vLLM Python adapter is a skeleton. Parser-based checks run as a separate
workflow; fluency evaluation and filtering remain future stages. Generation keeps
every returned outcome.

Requires Python 3.10 or newer. From this repository's root folder:

```bash
python -m pip install -e .
ctrl-fin-gen --help
```

The CLI uses `conditions/` and `experiments/` under the current working folder.
This initial version ships templates in the repository, so run the examples here.
`python -m ctrl_fin_gen` is an equivalent entry point.

The source packages follow the stages of the workflow:

| Package | Responsibility |
| --- | --- |
| `src/ctrl_fin_gen/` | Prepare/validate generation constraints, render prompts, generate texts, and save provider outcomes |
| [`src/linguistic_eval/`](src/linguistic_eval/README.md) | Parse saved texts, extract CoNLL-U features, and compare them with frozen constraints |
| Future sibling packages | Fluency/custom evaluation, filtering, and statistical analysis |

Generation ends with saved outcomes. An application-level coordinator can later
invoke evaluation after generation through the saved-artifact interfaces; the
generation package itself does not run evaluation. Linguistic evaluation provides
the independent `parse`, `extract`, and `check` stages. See its
[module guide](src/linguistic_eval/README.md).

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

## Generate texts with OpenAI

Set `OPENAI_API_KEY` in your environment before generation. Use a model available
to your account in place of `MODEL`. Model and reasoning-effort compatibility are
checked by the API; the CLI does not maintain a hard-coded model list.

```bash
ctrl-fin-gen generate science_pilot \
  --name openai_stream --backend openai --model MODEL \
  --max-length 2000 --samples 5 --stream
```

`--reasoning-effort low` is an OpenAI-owned option. Provider-specific flags and
validation live in that provider module. Without `--stream` or `--batch`, requests run
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

## Generate texts with Gemini

Install the optional Google SDK in your active Python/conda environment:

```bash
python -m pip install -e ".[google]"
export GEMINI_API_KEY="YOUR_API_KEY"
```

The `gemini` backend uses the Gemini Developer API with an API key. No Google
Cloud project, ADC login, or Cloud Storage bucket is needed. See the
[Google Gen AI SDK documentation](https://googleapis.github.io/python-genai/)
for API key setup and supported models.

```bash
ctrl-fin-gen generate science_pilot \
  --name gemini_stream --backend gemini --model MODEL \
  --max-length 2000 --samples 5 --stream
```

`GOOGLE_API_KEY` is also accepted and takes precedence if both standard key
variables are set. Use `--google-api-key-env MY_KEY_VARIABLE` to select a specific
variable. Only that variable's name is frozen in the manifest; the secret is
loaded at execution time and never saved in submission files. Resume uses the
same variable name. Omit `--stream` for ordinary sequential requests.

The earlier backend spelling `google_cloud` remains an alias for `gemini`.
Project-based settings are no longer supported; old project-based submissions
remain intact and require a fresh Gemini submission.

Optional provider settings:

| Flag | Purpose |
| --- | --- |
| `--google-api-key-env` | Environment variable holding the API key |
| `--google-temperature` | Sampling temperature, 0–2 |
| `--google-top-p` | Nucleus sampling, 0–1 |
| `--google-top-k` | Positive integer sampling limit |
| `--google-seed` | Generation seed, separate from vocabulary sampling |
| `--google-thinking-budget` | Thinking token budget; 0 disables it and -1 selects automatic budgeting where supported |
| `--google-thinking-level` | `minimal`, `low`, `medium`, or `high`; use instead of a thinking budget |
| `--google-timeout` | Per-request timeout in seconds; defaults to 120 |

Model-specific parameter support is checked by Google. Unspecified sampling and
thinking options use the service defaults. One request asks for one text; thought
parts stay in native artifacts and are excluded from the normalized text.

Batch jobs use the same key through Gemini's Files and Batch APIs:

```bash
ctrl-fin-gen generate science_pilot \
  --name gemini_batch --backend gemini --model MODEL \
  --max-length 2000 --samples 5 --batch
ctrl-fin-gen status science_pilot gemini_batch
ctrl-fin-gen collect science_pilot gemini_batch
```

The provider uploads keyed JSONL, submits the job, and downloads available output
when collected. Inputs and downloaded outputs are retained locally. Results are
matched by attempt key, including repeated prompts and out-of-order responses.
Collection is safe to repeat. Remote files are left under Gemini's file lifecycle;
the CLI does not delete them automatically.

## Results and recovery

Each submission lives in `experiments/EXPERIMENT/generations/NAME/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Saved model/settings, realized condition snapshot, source hashes |
| `requests.jsonl` | Backend-independent prompts, resolved conditions, configuration IDs, repetition indices, request IDs |
| `results.jsonl` | One normalized outcome per finished attempt, including response text and grouping IDs |
| `state.json` | Progress, active request, batch job ID |
| `attempts.jsonl` | Shared execution-attempt history, including retries |
| `events.jsonl` | Shared started, text-delta, and opaque checkpoint events |
| `provider/` | Provider-owned native requests, responses, streams, and batch files |

Streaming text is displayed on stderr; command summaries are JSON on stdout.
Every terminal response is saved before the next request. Partial responses,
refusals, and provider errors are retained with their status; they are not
filtered out. Unfinished text fragments remain in `events.jsonl`; native stream evidence is in `provider/`.

```bash
ctrl-fin-gen generate science_pilot --resume openai_stream
```

Resume reuses saved settings and prompts, skips finished samples, and retrieves
a known response ID where supported. It does not continue from an exact token.
OpenAI responses are submitted with `store: true` to support retrieval. Google
recovers terminal responses from local files or its saved stream journal; it
cannot retrieve an interrupted content request by response ID. SDK automatic
generation retries are disabled so attempts remain explicit. Google SDK automatic
continuation is also disabled: a length-limited response remains incomplete.

When a connection fails before a response/job ID is saved, resume first tries to
reconcile batch submissions using provider-owned metadata/checkpoints. If the outcome is still unknown,
it stops. To explicitly allow another attempt, which may create a duplicate
remote request and additional API charges:

```bash
ctrl-fin-gen generate science_pilot --resume openai_stream --retry-unknown
```

After a hard process crash, a submission's `.lock` file may remain. Remove it only
once you have confirmed that no process is writing to that submission. An
unfinished final JSONL line is backed up as `.partial` before recovery.

The shared submission schema is now **1.0**. Older prototype runs remain intact;
the new runner rejects them and requires a fresh submission, without migration
or automatic replay. Frozen request-plan hashes are checked before execution.

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

Tests use temporary experiment folders, fake providers, and the real OpenAI and
Google SDKs with in-memory HTTP transports. They require no credentials or paid requests.
They cover preparation, CLI commands, streaming, interruption recovery, batch
collection, and preservation of partial/error outputs. The project maintainer
has also confirmed successful live OpenAI generation after the provider refactor.
The automated suite remains offline and requires no API credentials.
Google tests run when `.[google]` is installed and are skipped otherwise.
Generation, resumable file upload/download, and batches all pass through the real
SDK with simulated HTTP. The project maintainer confirmed successful live Gemini
streaming generation on 2026-11-10. Live non-streaming, batch, and interruption
recovery checks remain pending. See [test coverage](tests/README.md).

See [conditions](conditions/README.md) for the three template tiers. Older planning
artifacts predate the runtime manifest format; create a fresh experiment and
submission for testing.

## Finnish parsing and deterministic checks

Trankit-large with **UD_Finnish-TDT** is available as an optional local parser.
It runs in a separate Python 3.11 environment, so the main project can keep its
newer Python and provider SDKs. From the repository root, with Python 3.11 and Git
installed:

```bash
python3.11 scripts/setup_trankit.py
```

This creates `.venv-trankit/`, installs pinned CPU dependencies and a pinned
Trankit source revision containing the Hugging Face download fix, downloads the
pinned Finnish-TDT and XLM-R-large models into `.cache/trankit-large-tdt/`, checks
their integrity, and parses a Finnish sample. Both directories are ignored by Git.
Allow roughly **4 GB of disk space** and several GB of available RAM. No GPU or
Hugging Face account is needed. Linux x86-64 / Python 3.11 is the verified worker
platform; the setup includes macOS/Windows paths but those platforms are untested.
On Windows, invoke the setup with `py -3.11 scripts/setup_trankit.py`.

Parse an experiment or one generation, then extract features and check its targets:

```bash
linguistic-eval parse --input-folder experiments/science_pilot
linguistic-eval extract --input evaluations/deterministic_checks/science_pilot/gemini_stream_1/conllus
linguistic-eval check --input evaluations/deterministic_checks/science_pilot/gemini_stream_1
```

Each generation receives `constraints.json`, `conllus/`, `feature_analysis/`, and
`constraint_satisfaction.jsonl`. Only completed attempts are evaluated, using their
frozen request constraints. Matching is strict, with `#` removed from lemmas for
checking; word lengths always count characters. Reruns replace outputs by default;
add `--only-new` to reuse unchanged entries. See the
[workflow guide](src/linguistic_eval/README.md) for output overrides and report semantics.

After setup, use the main project's Python environment as usual:

```bash
python -m linguistic_eval parse-text --text "Koulussa lapset lukevat kirjoja."
python -m linguistic_eval parse-text --input Finnish.txt --format conllu
```

The `linguistic-eval parse-text` entry point is equivalent after installing/reinstalling
the project. JSON output contains the original text, Trankit's native annotations
(including token spans, lemmas, POS, morphology, and dependencies), CoNLL-U, and
parser provenance: source/model revisions, resource SHA-256 hashes, library
versions, CPU settings, and seed. Output goes to stdout; model-loading messages
go to stderr. Redirect the JSON output to save an analyzed artifact. CoNLL-U-only
output omits the provenance envelope, so use JSON for reproducible checking.

For multiple texts, keep a worker alive and load the model only once:

```python
from linguistic_eval.trankit_support import TrankitParser

with TrankitParser() as parser:
    first = parser.parse("Koulussa lapset lukevat kirjoja.")
    second = parser.parse("Opettaja auttaa heitä.")
    print(first["analysis"]["sentences"][0]["tokens"])
```

Normal parsing is offline, verifies the resource hashes once per worker, and
fails explicitly if setup is missing or a resource changed. It does not contact
Trankit's old server or silently download a different model. Setup can be rerun;
downloads resume through Hugging Face's cache. Upstream's language name `finnish`
means **TDT**; `finnish-ftb` is a different model.

The default cache is relative to the current working folder. When running from
elsewhere, pass `cache_dir=` / `--cache-dir` or set
`CTRL_FIN_GEN_TRANKIT_CACHE` to an absolute path. A custom worker location can be
selected with `python=` / `--python` or `CTRL_FIN_GEN_TRANKIT_PYTHON`. Setup accepts
`--env-dir` and `--cache-dir`; use the same locations when parsing.

The worker uses CPU inference, one thread, fixed seeds, and PyTorch deterministic
algorithms. `threads=` / `--threads` changes the thread count and records it in
the output. These settings support repeatability in the pinned environment;
cross-platform bitwise equivalence is not promised. Raw compound markers
(such as `#`) and multi-word expansions remain in CoNLL-U; the extractor records
its normalization and counting policy separately. Whitespace-spanning word tokens
are split before tagging and lemmatization, with repairs recorded in provenance.
Word-length failures include the actual offending words and character counts. Parser errors raise
`ParserError`; they must not be interpreted as failed linguistic constraints.
Generation does not automatically parse or filter its outputs.

Run the optional real-model integration test after setup:

```bash
CTRL_FIN_GEN_TEST_TRANKIT=1 python -m unittest discover -s tests -p 'test_nlp.py' -v
```

It checks Finnish annotations and repeatability within a worker, after an
intervening text, and after restarting the worker. Ordinary tests do not download
or load NLP models. To check just the worker dependencies, run
`.venv-trankit/bin/python -m pip check`.

Resources come from [Trankit](https://github.com/nlp-uoregon/trankit), its
[official model mirror](https://huggingface.co/uonlp/trankit), and
[XLM-R-large](https://huggingface.co/FacebookAI/xlm-roberta-large).
Use their respective license/citation terms, including the
[Finnish-TDT treebank](https://universaldependencies.org/treebanks/fi_tdt/).


## Provider interface and request reuse

`providers/base.py` defines the provider interface. `generation.py` owns common
execution/progress only. Each provider owns native formatting, SDK calls,
status/error interpretation, and recovery. Register new implementations
in `providers/__init__.py`; no backend branch belongs in the shared runner.
OpenAI and Gemini are registered. The existing vLLM server adapter remains
available for development/testing but is not yet in the production registry.

`providers/vllm_python.py` provides a documented `VLLMPythonProvider` skeleton.
It implements the interface with explicit pending-work errors and remains
unregistered. Its engine configuration, chat-template handling, sampling options,
local inference grouping and cleanup will be chosen before implementation.
Local grouping belongs in `generate_many`; persistent cloud batch jobs are a
separate capability. No vLLM dependency or GPU is required to import the package.

Reuse a prepared request plan without resampling or changing prompts:

```bash
ctrl-fin-gen generate science_pilot --name another_model \
  --backend openai --model MODEL --max-length 2000 --samples 5 \
  --requests experiments/science_pilot/generations/openai_stream/requests.jsonl
```

The sample count must match the reused plan. Each request produces one intended
text; `configuration_id` and one-based `repetition_index` record planned grouping.
Condition-scope vocabulary shares a configuration across the requests;
request-scope sampling gives each draw its own configuration. An independent
texts-per-sampled-configuration control is future work. Retries keep the request
ID and assign a new attempt ID. All terminal attempts stay in `results.jsonl`;
progress counts requests separately. Resume preserves known terminal failures,
refusals and incomplete outputs as well as completed texts.
