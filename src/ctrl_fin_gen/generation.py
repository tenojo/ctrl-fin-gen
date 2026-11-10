"""Provider-independent submission execution.

Preparation freezes requests. Providers translate/execute/normalize them. This
module owns attempts, shared events, recovery policy and progress, and never
examines a provider's opaque checkpoint or native artifacts.
"""
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .models import (SCHEMA_VERSION, RESULT_STATUSES, JOB_TERMINAL, GenerationSettings,
                     GenerationRequest, GenerationAttempt, BatchJob)
from .prepare import load_experiment, prepare_requests
from .providers import create_provider
from .storage import (append_jsonl, content_hash, named_folder, read_json, read_jsonl,
                      repair_jsonl, submission_lock, timestamp, write_json, generation_folder, load_manifest)


def validate_generation_settings(settings):
    """Validate only common options; native options are validated by the provider."""
    if settings.mode not in {'immediate', 'stream', 'batch'}:
        raise ValueError('Mode must be immediate, stream, or batch.')
    if not isinstance(settings.model, str) or not settings.model.strip():
        raise ValueError('A model name is required.')
    value = settings.max_output_tokens
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('Token budget must be a positive integer.')


def start_generation(root, experiment, name, settings, samples, provider=None, on_text=None, requests=None):
    """Freeze a new run, then execute it. requests optionally reuses an exact shared plan."""
    validate_generation_settings(settings)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError('Sample count must be a positive integer.')
    folder = generation_folder(root, experiment, name)
    if folder.exists():
        raise FileExistsError(f'Generation already exists: {folder}. Use --resume.')
    inputs = load_experiment(root, experiment)
    plan = prepare_requests(inputs, samples) if requests is None else list(requests)
    if len(plan) != samples or len({r.request_id for r in plan}) != samples:
        raise ValueError('The request plan must contain the requested number of unique IDs.')
    for request in plan:
        if request.schema_version != SCHEMA_VERSION or request.repetition_index < 1:
            raise ValueError('Invalid shared request version or repetition index.')
    service = provider or create_provider(settings, folder / 'provider')
    try:
        service.validate_settings()
        if settings.mode == 'stream' and not service.streaming:
            raise ValueError('This provider does not support streaming.')
        if settings.mode == 'batch' and not service.batch_jobs:
            raise ValueError('This provider does not support background batch jobs.')
        folder.parent.mkdir(parents=True, exist_ok=True)
        folder.mkdir()
        (folder / 'provider').mkdir()
        for request in plan:
            append_jsonl(folder / 'requests.jsonl', asdict(request))
        write_json(folder / 'manifest.json', {
            'schema_version': SCHEMA_VERSION, 'submission_id': uuid4().hex,
            'experiment': experiment, 'name': name, 'created_at': timestamp(),
            'samples': samples, 'settings': asdict(settings), 'condition': inputs.template,
            'request_plan_sha256': content_hash(folder / 'requests.jsonl'),
        })
        write_json(folder / 'state.json', {'status': 'prepared', 'active': {},
                                         'batch_attempts': [], 'job': None, 'batch_checkpoint': {}})
        return resume_generation(root, experiment, name, provider=service, on_text=on_text)
    finally:
        if provider is None:
            service.close()


def save_result(folder, result):
    """Retain every terminal attempt; repeat collection is idempotent, conflicts fail."""
    row = asdict(result)
    manifest = load_manifest(folder)
    requests = {r['request_id']: r for r in read_jsonl(folder / 'requests.jsonl')}
    expected = requests.get(result.request_id)
    planned = [r for r in read_jsonl(folder / 'attempts.jsonl') if r['attempt_id'] == result.attempt_id]
    if not planned or planned[0]['request']['request_id'] != result.request_id:
        raise ValueError('Result does not identify a planned execution attempt.')
    if not expected or result.submission_id != manifest['submission_id']:
        raise ValueError('Result identifies an unknown request or submission.')
    if result.status not in RESULT_STATUSES or result.schema_version != SCHEMA_VERSION:
        raise ValueError('Result is not a supported terminal outcome.')
    if (result.configuration_id, result.repetition_index) != (expected['configuration_id'], expected['repetition_index']):
        raise ValueError('Result grouping differs from the frozen request.')
    key = ('submission_id', 'request_id', 'attempt_id')
    for old in read_jsonl(folder / 'results.jsonl'):
        if all(old[k] == row[k] for k in key):
            if old != row:
                raise ValueError('Conflicting terminal results for the same attempt.')
            return
    append_jsonl(folder / 'results.jsonl', row)
    append_jsonl(folder / 'attempts.jsonl', {**asdict(saved_attempt(planned[0])),
                                           'event': 'finished', 'at': timestamp()})


def progress(folder, state):
    """Count planned requests separately from finished attempts, then checkpoint."""
    results = read_jsonl(folder / 'results.jsonl')
    finished = {r['request_id'] for r in results}
    successful = {r['request_id'] for r in results if r['status'] == 'completed'}
    terminal_attempts = {r['attempt_id'] for r in results}
    state['active'] = {key: entry for key, entry in state['active'].items() if key not in terminal_attempts}
    total = load_manifest(folder)['samples']
    state.update(samples=total, finished_samples=len(finished), successful_samples=len(successful),
                 failed_samples=len(finished - successful), remaining_samples=total - len(finished),
                 finished_attempts=len(results), updated_at=timestamp())
    if len(finished) == total:
        state['status'] = 'completed' if len(successful) == total else 'completed_with_errors'
    write_json(folder / 'state.json', state)
    return state


def handle_event(folder, state, event, on_text=None):
    """Persist a canonical event before returning; checkpoint contents remain opaque."""
    if event.kind not in {'started', 'text_delta', 'checkpoint'}:
        raise ValueError(f'Unknown shared event: {event.kind}')
    append_jsonl(folder / 'events.jsonl', asdict(event))
    if event.kind == 'checkpoint':
        if event.attempt_id:
            state['active'][event.attempt_id]['checkpoint'] = event.checkpoint
        else:
            state['batch_checkpoint'] = event.checkpoint
        write_json(folder / 'state.json', state)
    if event.kind == 'text_delta' and on_text:
        on_text(event.text)


def pending_requests(folder):
    """Resume preserves all terminal outcomes, including failures and incomplete text."""
    finished = {r['request_id'] for r in read_jsonl(folder / 'results.jsonl')}
    return [GenerationRequest(**r) for r in read_jsonl(folder / 'requests.jsonl') if r['request_id'] not in finished]


def make_attempt(request, manifest):
    """Assign a fresh execution ID; request and configuration identities stay fixed."""
    return GenerationAttempt(request, manifest['submission_id'], uuid4().hex)


def saved_attempt(row):
    """Reconstruct a shared attempt from its saved record, without rerendering prompts."""
    return GenerationAttempt(GenerationRequest(**row['request']), row['submission_id'], row['attempt_id'])


def recover_requests(folder, state, service, retry_unknown):
    """Ask the provider for evidence, applying only shared recovery decisions."""
    waiting = False
    finished = {r['attempt_id'] for r in read_jsonl(folder / 'results.jsonl')}
    for attempt_id, entry in list(state['active'].items()):
        if attempt_id in finished:
            del state['active'][attempt_id]
            continue
        report = service.recover_request(saved_attempt(entry['attempt']), entry['checkpoint'])
        if report.outcome == 'finished' and report.result is not None:
            if report.result.attempt_id != attempt_id:
                raise ValueError('Recovery returned a result for another attempt.')
            save_result(folder, report.result)
        elif report.outcome == 'running':
            waiting = True
            continue
        elif report.outcome == 'safe_to_retry' or (report.outcome == 'unknown' and retry_unknown):
            append_jsonl(folder / 'attempts.jsonl', {**entry['attempt'], 'event': 'abandoned', 'at': timestamp()})
        else:
            raise ValueError('An interrupted request has an unknown remote outcome. Use --retry-unknown to authorize another attempt.')
        del state['active'][attempt_id]
    state['status'] = 'waiting_for_response' if waiting else 'running'
    progress(folder, state)
    return not waiting


def run_immediate(folder, manifest, state, service, retry_unknown, on_text):
    """Consume only shared results/events; a provider may group native inference work."""
    if not recover_requests(folder, state, service, retry_unknown):
        return state
    attempts = [make_attempt(r, manifest) for r in pending_requests(folder)]
    def events(event):
        # A prepared attempt not yet started is safe after a crash. Mark all work
        # as in-flight synchronously before a provider starts its native call.
        if event.kind == 'started':
            attempt = next(a for a in attempts if a.attempt_id == event.attempt_id)
            state['active'][attempt.attempt_id] = {'attempt': asdict(attempt), 'checkpoint': {}}
            append_jsonl(folder / 'attempts.jsonl', {**asdict(attempt), 'event': 'started', 'at': timestamp()})
            write_json(folder / 'state.json', state)
        handle_event(folder, state, event, on_text)
    try:
        for result in service.generate_many(attempts, events):
            if result.attempt_id not in state['active']:
                raise ValueError('Provider returned a result without starting the planned attempt.')
            save_result(folder, result)
            del state['active'][result.attempt_id]
            if on_text and manifest['settings']['mode'] == 'stream':
                on_text('\n')
            progress(folder, state)
    except (Exception, KeyboardInterrupt):
        state['status'] = 'interrupted'
        progress(folder, state)
        raise
    return progress(folder, state)


def run_batch(folder, manifest, state, service, retry_unknown, on_text=None):
    """Coordinate a background job without building or parsing native job files."""
    events = lambda event: handle_event(folder, state, event, on_text)
    if state['job']:
        job = service.get_batch_status(BatchJob(**state['job']))
    else:
        job = None
        if state['status'] == 'submitting':
            report = service.recover_batch(manifest['submission_id'], state['batch_checkpoint'])
            job = report.job
            if job is None and not (report.outcome == 'safe_to_retry' or retry_unknown):
                raise ValueError('Batch submission has an unknown outcome. Use --retry-unknown to authorize resubmission.')
        if job is None:
            if state['batch_attempts']:
                for old in state['batch_attempts']:
                    append_jsonl(folder / 'attempts.jsonl', {**old, 'event': 'abandoned', 'at': timestamp()})
                state['batch_attempts'] = []
            if not state['batch_attempts']:
                state['batch_attempts'] = [asdict(make_attempt(r, manifest)) for r in pending_requests(folder)]
            state['batch_checkpoint'] = {}
            for attempt in state['batch_attempts']:
                append_jsonl(folder / 'attempts.jsonl', {**attempt, 'event': 'started', 'at': timestamp()})
            state['status'] = 'submitting'
            write_json(folder / 'state.json', state)
            job = service.submit_batch([saved_attempt(a) for a in state['batch_attempts']], manifest['submission_id'], events)
    state.update(job=asdict(job), status=job.status)
    progress(folder, state)
    if job.status in JOB_TERMINAL:
        return collect_batch(folder, state, service, job)
    return state


def collect_batch(folder, state, service, job):
    """Persist normalized job outcomes; missing records remain visibly missing."""
    state.update(job=asdict(job), status=job.status, provider_status=job.status)
    for result in service.collect_batch(job, [saved_attempt(a) for a in state['batch_attempts']],
                                        lambda event: handle_event(folder, state, event)):
        save_result(folder, result)
    progress(folder, state)
    if job.status == 'completed' and state['remaining_samples']:
        state['status'] = 'missing_results'
        write_json(folder / 'state.json', state)
    return state


def resume_generation(root, experiment, name, provider=None, retry_unknown=False, on_text=None):
    """Continue frozen work under a lock; never change settings or retry terminal results."""
    folder = generation_folder(root, experiment, name)
    manifest = load_manifest(folder)
    with submission_lock(folder):
        for filename in ('results.jsonl', 'events.jsonl', 'attempts.jsonl'):
            repair_jsonl(folder / filename)
        state = read_json(folder / 'state.json')
        # Replay checkpoint events if interruption occurred between journal and state writes.
        for event in read_jsonl(folder / 'events.jsonl'):
            if event['kind'] == 'checkpoint':
                if event['attempt_id'] in state['active']:
                    state['active'][event['attempt_id']]['checkpoint'] = event['checkpoint']
                elif event['attempt_id'] is None:
                    state['batch_checkpoint'] = event['checkpoint']
        if not pending_requests(folder):
            return progress(folder, state)
        service = provider or create_provider(GenerationSettings(**manifest['settings']), folder / 'provider')
        try:
            if manifest['settings']['mode'] == 'batch':
                return run_batch(folder, manifest, state, service, retry_unknown, on_text)
            return run_immediate(folder, manifest, state, service, retry_unknown, on_text)
        finally:
            if provider is None:
                service.close()


def get_status(root, experiment, name, provider=None):
    """Read local progress and optionally query an existing shared job handle."""
    folder = generation_folder(root, experiment, name)
    manifest = load_manifest(folder)
    state = read_json(folder / 'state.json')
    if state['job']:
        service = provider or create_provider(GenerationSettings(**manifest['settings']), folder / 'provider')
        try:
            job = service.get_batch_status(BatchJob(**state['job']))
            state.update(provider_status=job.status, provider_counts=job.counts)
        finally:
            if provider is None:
                service.close()
    return state


def collect_results(root, experiment, name, provider=None):
    """Collect existing outcomes only; never submit jobs or generate replacement text."""
    folder = generation_folder(root, experiment, name)
    manifest = load_manifest(folder)
    with submission_lock(folder):
        repair_jsonl(folder / 'results.jsonl')
        state = read_json(folder / 'state.json')
        service = provider or create_provider(GenerationSettings(**manifest['settings']), folder / 'provider')
        try:
            if manifest['settings']['mode'] == 'batch':
                if not state['job']:
                    raise ValueError('No saved job. Use generate --resume to reconcile it.')
                return collect_batch(folder, state, service, service.get_batch_status(BatchJob(**state['job'])))
            for row in read_jsonl(folder / 'attempts.jsonl'):
                if row['event'] == 'started':
                    report = service.recover_request(saved_attempt(row), {})
                    if report.outcome == 'finished':
                        save_result(folder, report.result)
            return progress(folder, state)
        finally:
            if provider is None:
                service.close()
