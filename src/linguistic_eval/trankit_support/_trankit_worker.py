"""Private JSON-lines worker, run only in the pinned Python 3.11 environment."""

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

TOKENIZATION_POLICY = 'split_whitespace_word_forms_v1'


def split_whitespace_tokens(sentences):
    """Enforce written-word boundaries before tagging, not after predictions.

    Preserve sentence boundaries, source spans, and existing MWT expansions.
    Renumber every word/MWT range in an affected sentence so the tagger can
    predict fresh heads and lemmas for the corrected segmentation.
    """
    repairs = []
    for sentence in sentences:
        tokens = sentence['tokens']
        if not any(len(word['text'].split()) > 1
                   for token in tokens for word in token.get('expanded', [token])):
            continue
        next_id, corrected = 1, []

        def split_word(word):
            parts = list(re.finditer(r'\S+', word['text']))
            if len(parts) > 1:
                repairs.append({'sentence_id': sentence['id'], 'form': word['text'],
                                'parts': [part.group() for part in parts]})
            result = []
            for part in parts:
                item = {**word, 'text': part.group()}
                for key in ('span', 'dspan'):
                    if key in word:
                        start = word[key][0]
                        item[key] = (start + part.start(), start + part.end())
                result.append(item)
            if not result:
                raise ValueError('Tokenizer returned an empty word.')
            return result

        for token in tokens:
            words = [part for word in token.get('expanded', [token]) for part in split_word(word)]
            for offset, word in enumerate(words):
                word['id'] = next_id + offset
            if 'expanded' in token:
                corrected.append({**token, 'id': (next_id, next_id + len(words) - 1), 'expanded': words})
            else:
                corrected.extend(words)
            next_id += len(words)
        sentence['tokens'] = corrected
    return repairs


def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--cache-dir', type=Path, required=True)
    args.add_argument('--threads', type=int, required=True)
    options = args.parse_args()
    try:
        from .trankit_resources import EMBEDDING, LANGUAGE, verify_resources

        resources = verify_resources(options.cache_dir)
        with redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from trankit import Pipeline, trankit2conllu

            class FinnishPipeline(Pipeline):
                # Trankit's pinned implementation calls this before POS/dependency
                # prediction and lemmatization. Keep its MWT expansion machinery.
                def _tokenize_doc(self, in_doc):
                    sentences = super()._tokenize_doc(in_doc)
                    self.whitespace_repairs = split_whitespace_tokens(sentences)
                    return sentences

            torch.set_num_threads(options.threads)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            np.random.seed(0)
            torch.use_deterministic_algorithms(True)
            pipeline = FinnishPipeline(lang=LANGUAGE, embedding=EMBEDDING,
                                       cache_dir=str(options.cache_dir), gpu=False)
        metadata = {
            'resources': resources, 'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in (
                'trankit', 'torch', 'adapters', 'transformers', 'numpy',
                'huggingface-hub', 'sentencepiece', 'tokenizers',
            )},
            'device': 'cpu', 'threads': options.threads, 'seed': 0,
            'deterministic_algorithms': True,
            'tokenization_policy': TOKENIZATION_POLICY,
        }
        send({'ready': metadata})
    except Exception as error:
        send({'error': f'{type(error).__name__}: {error}'})
        return 1

    for line in sys.stdin:
        try:
            text = json.loads(line)['text']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Text must be a non-empty string.')
            with redirect_stdout(sys.stderr), torch.inference_mode():
                analysis = pipeline(text)
                conllu = trankit2conllu(analysis)
            if conllu is None:
                raise RuntimeError('Trankit could not produce CoNLL-U.')
            send({'result': {'schema_version': '1.0', 'text': text,
                             'parser': {**metadata, 'tokenization_repairs': pipeline.whitespace_repairs},
                             'analysis': analysis, 'conllu': conllu}})
        except Exception as error:
            send({'error': f'{type(error).__name__}: {error}'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
