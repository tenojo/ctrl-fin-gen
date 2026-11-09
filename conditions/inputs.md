# Reusable inputs

Pass a condition file to `create --condition FILE.json`. Omitted values use system
template defaults. CLI overrides apply afterward. For example:

```json
{
  "audience": {"description": "Peruskoulun 3. luokan oppilaat"},
  "prompt": "Kirjoita kertomus metsäretkestä.",
  "constraints": {
    "vocabulary_constraints": {
      "selection": {"mode": "inline", "items": ["metsä", "polku"]}
    },
    "length_constraints": {
      "number_of_sentences": {"min": 4, "max": 5}
    },
    "linguistic_constraints": {
      "selection": {
        "mode": "inline",
        "items": [{"control": "Case=Gen", "occurrences": {"min": 1, "max": null}}]
      }
    }
  }
}
```

A numeric feature instead uses `{"control": "MaxTreeDepth", "target": {"min": 2,
"max": 4}}`; a token-proportion target uses `proportion` with bounds from 0 to 1.
These are prompt instructions; no parser-backed analysis runs during generation.

Vocabulary selection supports three forms:

```json
{"selection": {"mode": "inline", "items": ["koulu", "kirja"]}}
```

```json
{"selection": {"mode": "file", "path": "lemmas.txt"}}
```

```json
{"selection": {"mode": "sample", "pool": "pool.json", "count": 3, "seed": 42, "scope": "request"}}
```

File paths in a condition file resolve relative to that file. CLI paths resolve
relative to the current working folder.

A CSV pool needs a JSON import specification so column names, encoding, and
preprocessing are explicit. For example, save the following as `pool.json` and
use it with `--vocabulary-pool pool.json` or `--lemmas-file pool.json`:

```json
{
  "path": "science_words.csv",
  "encoding": "utf-8",
  "delimiter": ";",
  "column": "lemma",
  "where": {"grade": "3"},
  "remove_compound_markers": true,
  "duplicate_policy": "unique"
}
```

`path` resolves relative to the import specification. `where` keeps rows matching
all supplied values. `remove_compound_markers` removes `#`; `unique` keeps the
first occurrence after Unicode normalization, ignoring case for duplicates.
Those three settings are optional. By default rows are unfiltered, compound
markers are retained, and duplicate lemmas raise an error. Source paths, hashes,
import settings, and sampling seeds are recorded in the realized template.

