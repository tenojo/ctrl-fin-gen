"""Explicit model provisioning; normal parsing never downloads resources."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

TRANKIT_COMMIT = '54e863327391262cf72f6adc1b0ff104e972a1dc'
MODEL_REVISION = '1add34e3909949dbb98523ed6cfcb4e9393e2c8e'
ENCODER_REVISION = 'c23d21b0620b635a76227c604d44e43a9f0ee389'
EMBEDDING = 'xlm-roberta-large'
LANGUAGE = 'finnish'  # upstream maps this to UD_Finnish-TDT; not finnish-ftb
ENCODER_FILES = ('config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model',
                 'tokenizer.json', 'tokenizer_config.json')
TASK_FILES = ('finnish.tokenizer.mdl', 'finnish.tagger.mdl',
              'finnish.vocabs.json', 'finnish_lemmatizer.pt', 'finnish_mwt_expander.pt')


def default_cache():
    return Path(os.environ.get('CTRL_FIN_GEN_TRANKIT_CACHE', '.cache/trankit-large-tdt')).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_resources(cache):
    """Reject missing/modified resources before upstream can try its downloader."""
    cache = Path(cache)
    manifest = json.loads((cache / 'resources.json').read_text(encoding='utf-8'))
    if (manifest['trankit_commit'], manifest['model_revision'], manifest['encoder_revision']) != (
            TRANKIT_COMMIT, MODEL_REVISION, ENCODER_REVISION):
        raise ValueError('Parser resources have different revisions; rerun scripts/setup_trankit.py.')
    expected = [f'{EMBEDDING}/finnish/{name}' for name in TASK_FILES]
    expected += [f'{EMBEDDING}/models--{EMBEDDING}/snapshots/{ENCODER_REVISION}/{name}'
                 for name in ENCODER_FILES]
    if set(manifest['sha256']) != set(expected):
        raise ValueError('Incomplete parser resource manifest; rerun scripts/setup_trankit.py.')
    for name in expected:
        if sha256(cache / name) != manifest['sha256'][name]:
            raise ValueError(f'Parser resource checksum mismatch: {name}')
    ref = cache / EMBEDDING / f'models--{EMBEDDING}' / 'refs/main'
    if ref.read_text().strip() != ENCODER_REVISION:
        raise ValueError('Unexpected encoder revision in cache.')
    if not (cache / EMBEDDING / LANGUAGE / 'finnish.downloaded').is_file():
        raise ValueError('Missing Trankit completion marker; rerun setup.')
    return manifest


def download_resources(cache):
    from filelock import FileLock
    from huggingface_hub import hf_hub_download, snapshot_download

    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / '.setup.lock')):
        # A successful repeat invocation can work offline as well.
        try:
            return verify_resources(cache)
        except (OSError, ValueError, KeyError):
            pass
        # A recorded-but-invalid installation must be fetched again, rather than
        # blessing modified files already present in Hugging Face's local cache.
        force_download = (cache / 'resources.json').exists()
        encoder_cache = cache / EMBEDDING
        snapshot = Path(snapshot_download(
            EMBEDDING, revision=ENCODER_REVISION, cache_dir=str(encoder_cache),
            allow_patterns=list(ENCODER_FILES), max_workers=2, force_download=force_download,
        ))
        # Upstream hardcodes the model ID and main revision. Point its local cache
        # at our pinned snapshot; workers use offline mode so main cannot drift.
        ref = snapshot.parent.parent / 'refs/main'
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(ENCODER_REVISION, encoding='utf-8')
        # Avoid Transformers' first-import migration warning on a fresh HF cache.
        (encoder_cache / 'version.txt').write_text('1', encoding='utf-8')
        archive = hf_hub_download(
            'uonlp/trankit', f'models/v1.0.0/{EMBEDDING}/finnish.zip',
            revision=MODEL_REVISION, cache_dir=str(cache / 'downloads'),
            force_download=force_download,
        )
        target = encoder_cache / LANGUAGE
        # Extract only the required, flat files into a staging directory. The
        # completion marker is written after extraction, never on a failed fetch.
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            with zipfile.ZipFile(archive) as zipped:
                for name in TASK_FILES:
                    Path(tmp, name).write_bytes(zipped.read(name))
            target.mkdir(parents=True, exist_ok=True)
            for name in TASK_FILES:
                os.replace(Path(tmp, name), target / name)
        (target / 'finnish.downloaded').touch()
        files = [target / name for name in TASK_FILES] + [snapshot / name for name in ENCODER_FILES]
        manifest = {
            'trankit_commit': TRANKIT_COMMIT, 'model_revision': MODEL_REVISION,
            'encoder_revision': ENCODER_REVISION, 'language': LANGUAGE,
            'treebank': 'UD_Finnish-TDT', 'embedding': EMBEDDING,
            'sha256': {str(path.relative_to(cache)): sha256(path) for path in files},
        }
        temporary = cache / 'resources.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, cache / 'resources.json')
        return verify_resources(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=default_cache())
    args = parser.parse_args()
    print(json.dumps(download_resources(args.cache_dir), indent=2))


if __name__ == '__main__':
    main()
