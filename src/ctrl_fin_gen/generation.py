"""Submission orchestration. Provider and file operations stay in their modules."""

import json
import os
from pathlib import Path
from uuid import uuid4

from openai import OpenAIError

from .constraints.vocabulary import vocabulary_for_request
from .prepare import load_experiment
from .prompts import build_messages
from .providers.openai import (
    OpenAIProvider, TERMINAL_BATCHES, TERMINAL_RESPONSES, format_response,
)
from .storage import (
    append_jsonl, content_hash, named_folder, read_json, read_jsonl,
    repair_jsonl, submission_lock, timestamp, write_json,
)


def generation_folder(root, experiment, name):
    return named_folder(named_folder(Path(root) / 'experiments', experiment) / 'generations', name)


def load_manifest(folder):
    manifest = read_json(folder / 'manifest.json')
    if not isinstance(manifest.get('settings'), dict):
        raise ValueError(
            'This folder does not contain a runtime submission. '
            'Create a new named generation; static planning examples cannot be resumed.'
        )
    return manifest


def save_result(folder, sample_id, body, attempt_id):
    requests = {row['sample_id'] for row in read_jsonl(folder / 'requests.jsonl')}
    if sample_id not in requests:
        raise ValueError(f'Received an unknown sample ID: {sample_id}')
    if sample_id not in {row['sample_id'] for row in read_jsonl(folder / 'results.jsonl')}:
        append_jsonl(folder / 'results.jsonl', format_response(sample_id, body, attempt_id))


def progress(folder, state):
    results = read_jsonl(folder / 'results.jsonl')
    state['finished_samples'] = len({row['sample_id'] for row in results})
    state['successful_samples'] = sum(row['status'] == 'completed' for row in results)
    state['failed_samples'] = sum(row['status'] != 'completed' for row in results)
    total = read_json(folder / 'manifest.json')['samples']
    state['samples'] = total
    state['remaining_samples'] = total - state['finished_samples']
    if state['finished_samples'] == total:
        state['status'] = 'completed' if state['failed_samples'] == 0 else 'completed_with_errors'
        state['active_request'] = None
    state['updated_at'] = timestamp()
    write_json(folder / 'state.json', state)
    return state


def start_generation(root, experiment, name, settings, provider=None, on_text=None):
    if settings['backend'] != 'openai':
        raise ValueError('Only the OpenAI backend is implemented.')
    for key in ('samples', 'max_output_tokens'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Sample count and token budget must be positive integers.')
    if not isinstance(settings.get('model'), str) or not settings['model'].strip():
        raise ValueError('A model name is required.')
    if settings.get('stream') and settings.get('batch'):
        raise ValueError('Streaming and batching are mutually exclusive.')
    folder = generation_folder(root, experiment, name)
    if folder.exists():
        raise FileExistsError(f'Generation already exists: {folder}. Use --resume to continue it.')
    inputs = load_experiment(root, experiment)
    submission_id = uuid4().hex
    requests = []
    for index in range(settings['samples']):
        vocabulary_settings = inputs.template['constraints'].get('vocabulary_constraints', {'items': []})
        vocabulary = vocabulary_for_request(vocabulary_settings, index)
        sample_id = f'{submission_id}-{index + 1:06d}'
        body = {
            'model': settings['model'], 'input': build_messages(inputs, vocabulary),
            'max_output_tokens': settings['max_output_tokens'], 'store': True,
            'metadata': {'sample_id': sample_id, 'submission_id': submission_id},
        }
        if settings.get('reasoning_effort') is not None:
            body['reasoning'] = {'effort': settings['reasoning_effort']}
        requests.append({'sample_id': sample_id, 'vocabulary': vocabulary, 'body': body})
    provider = provider if provider is not None else OpenAIProvider(base_url=settings.get('base_url'))
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()  # Reserve before any request or batch submission.
    (folder / 'raw').mkdir()
    template_path = Path(root) / 'experiments' / experiment / 'template/template.json'
    manifest = {
        'artifact_type': 'generation_submission', 'submission_id': submission_id,
        'experiment': experiment, 'name': name, 'created_at': timestamp(),
        'samples': settings['samples'], 'settings': settings,
        'template_path': str(template_path.resolve()),
        'template_sha256': content_hash(template_path), 'condition': inputs.template,
    }
    write_json(folder / 'manifest.json', manifest)
    with (folder / 'requests.jsonl').open('w', encoding='utf-8') as output:
        for request in requests:
            output.write(json.dumps(request, ensure_ascii=False) + '\n')
        output.flush()
        os.fsync(output.fileno())
    write_json(folder / 'state.json', {'status': 'prepared', 'provider_job_id': None, 'active_request': None})
    return resume_generation(root, experiment, name, provider=provider, on_text=on_text)


def recover_request(folder, state, provider, retry_unknown):
    active = state.get('active_request')
    if not active:
        return True
    sample_id = active['sample_id']
    if sample_id in {row['sample_id'] for row in read_jsonl(folder / 'results.jsonl')}:
        state['active_request'] = None
        return True
    raw = folder / 'raw' / f"{sample_id}-{active['attempt_id']}.json"
    body = read_json(raw) if raw.exists() else None
    if body is None:
        # An event can be durable even when the following checkpoint was interrupted.
        for row in read_jsonl(folder / 'raw/events.jsonl'):
            if row['attempt_id'] == active['attempt_id'] and 'response' in row['event']:
                body = row['event']['response']
                active['response_id'] = body.get('id')
    if (not body or body.get('status') not in TERMINAL_RESPONSES) and active.get('response_id'):
        try:
            body = provider.retrieve_response(active['response_id'])
        except OpenAIError:
            if not retry_unknown:
                raise
            body = None
        else:
            write_json(raw, body)
    if not body and (active.get('rejected') or retry_unknown):
        append_jsonl(folder / 'attempts.jsonl', {**active, 'event': 'abandoned', 'at': timestamp()})
        state['active_request'] = None
        return True
    if not body:
        raise ValueError(
            'An interrupted request has an unknown remote outcome. '
            'Use --retry-unknown only if you want to regenerate that sample.'
        )
    if body.get('status') not in TERMINAL_RESPONSES:
        state['status'] = 'waiting_for_response'
        return False
    save_result(folder, sample_id, body, active['attempt_id'])
    state['active_request'] = None
    return True


def run_immediate(folder, manifest, state, provider, retry_unknown, on_text):
    if not recover_request(folder, state, provider, retry_unknown):
        return progress(folder, state)
    finished = {row['sample_id'] for row in read_jsonl(folder / 'results.jsonl')}
    for request in read_jsonl(folder / 'requests.jsonl'):
        sample_id = request['sample_id']
        if sample_id in finished:
            continue
        active = {'sample_id': sample_id, 'attempt_id': uuid4().hex, 'response_id': None}
        state.update({'status': 'running', 'active_request': active})
        write_json(folder / 'state.json', state)
        append_jsonl(folder / 'attempts.jsonl', {**active, 'event': 'started', 'at': timestamp()})

        def on_event(event):
            append_jsonl(folder / 'raw/events.jsonl', {
                'sample_id': sample_id, 'attempt_id': active['attempt_id'], 'event': event,
            })
            if 'response' in event:
                active['response_id'] = event['response'].get('id')
                write_json(folder / 'state.json', state)
                if event['response'].get('status') in TERMINAL_RESPONSES:
                    write_json(folder / 'raw' / f"{sample_id}-{active['attempt_id']}.json", event['response'])
            if event.get('type') == 'response.output_text.delta' and on_text:
                on_text(event['delta'])

        try:
            body = provider.generate(request['body'], manifest['settings'].get('stream', False), on_event)
            if body.get('status') not in TERMINAL_RESPONSES:
                raise RuntimeError('The provider did not return a final response.')
            write_json(folder / 'raw' / f"{sample_id}-{active['attempt_id']}.json", body)
            save_result(folder, sample_id, body, active['attempt_id'])
            if on_text and manifest['settings'].get('stream'):
                on_text('\n')
            append_jsonl(folder / 'attempts.jsonl', {**active, 'event': 'finished', 'at': timestamp()})
            state['active_request'] = None
            progress(folder, state)
        except (Exception, KeyboardInterrupt) as error:
            active['rejected'] = getattr(error, 'status_code', None) in (400, 401, 403, 404, 422, 429)
            append_jsonl(folder / 'attempts.jsonl', {
                **active, 'event': 'interrupted', 'error': str(error), 'at': timestamp(),
                'http_status': getattr(error, 'status_code', None),
                'provider_error': getattr(error, 'body', None),
            })
            state['status'] = 'interrupted'
            progress(folder, state)
            if isinstance(error, KeyboardInterrupt):
                raise
            raise RuntimeError(
                f'Generation interrupted: {error}. Resume the saved submission to continue.'
            ) from error
    return progress(folder, state)


def batch_job(folder, manifest, state, provider, retry_unknown=False):
    if state.get('provider_job_id'):
        return provider.retrieve_batch(state['provider_job_id'])
    if state['status'] == 'submitting':
        job = provider.find_batch(manifest['submission_id'])
        if job is not None:
            state['provider_job_id'] = job['id']
            write_json(folder / 'state.json', state)
            return job
        if not retry_unknown:
            raise ValueError(
                'Batch submission has an unknown outcome. No matching job was found; '
                'use --retry-unknown to submit again explicitly.'
            )
    path = folder / 'batch_input.jsonl'
    with path.open('w', encoding='utf-8') as output:
        for request in read_jsonl(folder / 'requests.jsonl'):
            entry = {
                'custom_id': request['sample_id'], 'method': 'POST',
                'url': '/v1/responses', 'body': request['body'],
            }
            output.write(json.dumps(entry, ensure_ascii=False) + '\n')
    state['status'] = 'submitting'
    write_json(folder / 'state.json', state)
    job = provider.submit_batch(path, manifest['submission_id'])
    state['provider_job_id'] = job['id']
    state['status'] = job['status']
    write_json(folder / 'state.json', state)
    write_json(folder / 'raw/batch_job.json', job)
    return job


def collect_batch(folder, job, state, provider):
    write_json(folder / 'raw/batch_job.json', job)
    for key in ('output_file_id', 'error_file_id'):
        if not job.get(key):
            continue
        text = provider.download(job[key])
        (folder / 'raw' / (key + '.jsonl')).write_text(text, encoding='utf-8')
        for line in text.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            response = entry.get('response') or {}
            body = response.get('body') or {'status': 'failed', 'error': entry.get('error') or response}
            if entry.get('error'):
                body['status'] = 'failed'
            save_result(folder, entry['custom_id'], body, 'batch')
    state['status'] = job['status']
    state['provider_job_id'] = job['id']
    state['provider_status'] = job['status']
    received = {row['sample_id'] for row in read_jsonl(folder / 'results.jsonl')}
    if job['status'] == 'completed' and len(received) < read_json(folder / 'manifest.json')['samples']:
        state['status'] = 'missing_results'
    return progress(folder, state)


def resume_generation(root, experiment, name, provider=None, retry_unknown=False, on_text=None):
    folder = generation_folder(root, experiment, name)
    manifest = load_manifest(folder)
    if manifest['settings']['backend'] != 'openai':
        raise ValueError('Only the OpenAI backend is implemented.')
    with submission_lock(folder):
        for path in (folder / 'results.jsonl', folder / 'attempts.jsonl', folder / 'raw/events.jsonl'):
            repair_jsonl(path)
        state = read_json(folder / 'state.json')
        if len({row['sample_id'] for row in read_jsonl(folder / 'results.jsonl')}) == manifest['samples']:
            return progress(folder, state)
        provider = provider if provider is not None else OpenAIProvider(base_url=manifest['settings'].get('base_url'))
        if manifest['settings'].get('batch'):
            job = batch_job(folder, manifest, state, provider, retry_unknown)
            if job['status'] in TERMINAL_BATCHES:
                return collect_batch(folder, job, state, provider)
            return progress(folder, {**state, 'status': job['status']})
        return run_immediate(folder, manifest, state, provider, retry_unknown, on_text)


def get_status(root, experiment, name, provider=None):
    folder = generation_folder(root, experiment, name)
    manifest = load_manifest(folder)
    state = read_json(folder / 'state.json')
    state['finished_samples'] = len({row['sample_id'] for row in read_jsonl(folder / 'results.jsonl')})
    state['samples'] = manifest['samples']
    if manifest['settings'].get('batch') and state.get('provider_job_id'):
        provider = provider if provider is not None else OpenAIProvider(base_url=manifest['settings'].get('base_url'))
        job = provider.retrieve_batch(state['provider_job_id'])
        state.update({'provider_status': job['status'], 'provider_counts': job.get('request_counts')})
    return state


def collect_results(root, experiment, name, provider=None):
    folder = generation_folder(root, experiment, name)
    manifest = load_manifest(folder)
    with submission_lock(folder):
        repair_jsonl(folder / 'results.jsonl')
        state = read_json(folder / 'state.json')
        if manifest['settings'].get('batch'):
            provider = provider if provider is not None else OpenAIProvider(base_url=manifest['settings'].get('base_url'))
            if not state.get('provider_job_id'):
                raise ValueError('No saved batch job ID. Use generate --resume to reconcile submission first.')
            return collect_batch(folder, provider.retrieve_batch(state['provider_job_id']), state, provider)
        for path in sorted((folder / 'raw').glob('*.json')):
            body = read_json(path)
            if body.get('status') not in TERMINAL_RESPONSES:
                continue
            sample_id, attempt_id = path.stem.rsplit('-', 1)
            save_result(folder, sample_id, body, attempt_id)
        return progress(folder, state)
