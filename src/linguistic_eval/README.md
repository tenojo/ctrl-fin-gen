# Parser-based deterministic checks

This package consumes saved generation files without importing `ctrl_fin_gen`.
The three stages run independently. Generation itself never calls evaluation.

| Module | Responsibility |
| --- | --- |
| `parser_process.py` | Detect experiment/generation input, select completed results, save constraints and CoNLL-U |
| `feature_extractor.py` | Read CoNLL-U and measure lemma, morphological-feature, dependency-relation, POS, and length counts |
| `parser_checks.py` | Compare measurements with per-text targets and write satisfaction reports |
| `trankit_support/` | Configure the offline Trankit-large/Finnish-TDT backend |
| `args.py`, `cli.py` | Arguments, dispatch, and short stage summaries |

## Run the workflow

Install the project with `python -m pip install -e .` and configure the parser
once with `python3.11 scripts/setup_trankit.py`. From the repository root:

```bash
linguistic-eval parse --input-folder experiments/science_pilot
linguistic-eval extract --input evaluations/deterministic_checks/science_pilot/gemini_stream_1/conllus
linguistic-eval check --input evaluations/deterministic_checks/science_pilot/gemini_stream_1
```

`python -m linguistic_eval` is equivalent to `linguistic-eval`. Supply one
`experiments/X/generations/Y` folder to parse just that generation. Detection uses
`X/template/template.json`, not the spelling of the supplied experiment name.
Only completed results are selected; each attempt remains a separate text.
Failed, refused, cancelled, and incomplete results are excluded and counted in
`skipped_results`. A completed result with empty text produces a recorded parser
error rather than an empty successful analysis.

The default output for experiment X is `evaluations/deterministic_checks/X/`,
with one subfolder per generation. For generation Y alone it is
`evaluations/deterministic_checks/X/Y/`. The default evaluation root is beside
`experiments/`, so absolute inputs work from other working directories too.
`parse --output-folder PATH` replaces the output root at the selected scope:
experiment input writes `PATH/Y/`; single-generation input writes directly in PATH.
The source generation files are never modified.

```text
Y/
    constraints.json
    conllus/
        <text_id>.conllu
        <text_id>.json
    feature_analysis/
        <text_id>.json
    constraint_satisfaction.jsonl
```

`constraints.json` contains one `texts` entry per completed attempt: its original
text, request/attempt/configuration/repetition IDs, text hash, and frozen targets.
Use the saved request's conditions, including its sampled lemma list; never
reconstruct targets from the current experiment template. The older prototype's
`sample_id`/manifest-condition format is also supported. Stable text IDs are
hashes of request/attempt IDs and are used only as filenames and join keys.
CoNLL-U sidecars record parser/resource provenance, hashes, and parse status/errors.
The Finnish backend enforces whitespace boundaries before POS/dependency tagging
and lemmatization. If its tokenizer fuses several written words into one word
token, the components receive fresh predictions. Sentence boundaries, source
spans, and existing multi-word-token expansions are preserved. Sidecars record
the policy and any repaired forms; no word-specific exceptions are used.

`extract --input CONLLUS` processes every `.conllu` in that folder. Its default
output is the sibling `feature_analysis/`; change it with `--output-folder`.
Feature JSON records counts, extraction version, counting policy, input hash,
and extraction status/errors. No parser or model is loaded during extraction.

`check --input Y` finds `constraints.json` and `feature_analysis/` automatically.
Optional overrides are `--constraints FILE`, `--feature-analysis FOLDER`, and
`--output FILE`. By default it writes `Y/constraint_satisfaction.jsonl`.

## Matching and reports

- Lemmas match whole strings, case-sensitively, after removing `#` from both
  parsed and requested lemmas. No stemming, substring matching, or case folding.
- Morphological features match complete FEATS entries. `Case=Nom` does not match
  `Case=Acc,Nom`. Dependency relations and subtypes match exactly: `acl` does not
  match `acl:relcl`. POS uses UPOS; the three existing `*_headed_phrase` controls
  count roots of that POS.
- Presence, occurrence bounds, and token-proportion bounds use the saved target.
  Proportions divide by all integer-ID word rows, including punctuation/symbols.
- Sentence count uses CoNLL-U sentence boundaries. Word count excludes UPOS
  `PUNCT` and `SYM`. Multi-word-token headers and empty nodes never count again.
  Word FORMs are also split on whitespace when measuring lengths, so imported
  fused forms cannot inflate the maximum. Maximum word length is the number of
  Unicode characters in each component, including internal hyphens/digits, and
  excludes punctuation/symbol tokens. Linguistic checks on unrepaired fused word
  tokens are unavailable until re-parsing; their component lemmas are never guessed.
- Complex constructions, tree-depth metrics, and frequency/readability measures
  are not implemented in this first version. Requested unsupported controls
  receive explicit failure reasons, never a successful zero measurement.

Each report line retains its source IDs and contains `success` (0/1),
`satisfied_groups`, `satisfied_features`, `failed_groups`, `failed_features`, and
`failure_reasons`. Each failure records `feature`, `target`, and `realized_value`,
plus its group and an explanation when a measurement is unavailable/unsupported.
Word-length failures also list the offending written words and their character
counts under `words`, so the result can be traced back to the text.
Group names match generation's constraint categories. A group is satisfied only
if all its selected constraints pass; empty groups are omitted. Individual
features may pass within a group that fails overall. With no selected constraints,
valid analysis passes vacuously.

Missing parser output, missing annotations, malformed CoNLL-U, and stale feature
files cannot count as successful checks. Reports retain these errors explicitly
and use null rather than invented measurements. Operational errors give the CLI
exit status 1; ordinary target violations still give exit status 0 because the
check itself ran successfully.

## Replacement and incremental runs

All three stages replace their own corresponding entries by default. JSONL
reports are rewritten, not appended, so reruns do not create duplicate lines.
Add `--only-new` to reuse successful entries with unchanged inputs. Changed text,
CoNLL-U, constraints, or processing version is recomputed at the appropriate stage. Unchanged saved
parses can be reused without loading Trankit, which is useful for whole experiments.
Parser/extraction errors are retried. If results are removed or cease being
completed, their previously indexed parse files and report rows are removed.
Unreferenced feature files may remain but are not included in reports.

Run stages in order after changing inputs. `check` verifies the analysis hashes
and reports stale/missing analysis rather than using it. To apply a different
parser/model to existing texts, use the default replacement mode. These commands
are for sequential use; concurrent writers to the same evaluation folder are
not supported.

## Inspect one text or use Python

```bash
linguistic-eval parse-text --text "Koulussa lapset lukevat kirjoja."
linguistic-eval parse-text --input Finnish.txt --format conllu
```

```python
from linguistic_eval.trankit_support import TrankitParser
from linguistic_eval.parser_process import parse_folder
from linguistic_eval.feature_extractor import extract_folder
from linguistic_eval.parser_checks import check_folder

with TrankitParser() as parser:
    outputs = parse_folder("experiments/science_pilot", parser=parser, only_new=True)

for output in outputs:
    folder = output["output_folder"]
    extract_folder(f"{folder}/conllus", only_new=True)
    check_folder(folder, only_new=True)
```

Fluency/custom evaluation, filtering, and statistical analysis remain separate
future stages. An application-level coordinator can invoke generation followed
by these stages; orchestration does not belong in the generation package.
