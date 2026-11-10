"""OpenAI-only translation, SDK operations, native artifacts, and recovery.

The runner never sees Responses events, Batch envelopes, file IDs, HTTP status
codes, or SDK exceptions. This provider maps them to the shared contract.
"""
import json

from .base import Provider, ProviderError
from .responses import ResponsesProvider
from ..models import BatchJob, GenerationResult, ProviderEvent, RecoveryReport
from ..storage import append_jsonl, read_json, read_jsonl, write_json

TERMINAL = {'completed', 'incomplete', 'failed', 'cancelled'}


class OpenAIProvider(ResponsesProvider):
    """Responses and Batch client, initialized lazily and reused within a session."""
    streaming = True
    batch_jobs = True

    @property
    def client(self):
        """Load the optional SDK only when a remote operation is needed."""
        if self._client is None:
            from openai import OpenAI, OpenAIError
            try:
                self._client = OpenAI(
                    base_url=self.settings.provider_options.get('base_url'), max_retries=0,
                )
            except OpenAIError as error:
                raise ProviderError(str(error)) from error
        return self._client

    @classmethod
    def add_cli_arguments(cls, parser):
        """Keep OpenAI-specific flag definitions outside the shared CLI."""
        group = parser.add_argument_group('OpenAI options')
        group.add_argument('--reasoning-effort', help='Model-supported reasoning effort.')
        group.add_argument('--base-url', help='Override the OpenAI API address.')

    @classmethod
    def options_from_args(cls, args):
        """Extract only explicitly supplied native options."""
        return {key: getattr(args, key) for key in ('reasoning_effort', 'base_url')
                if getattr(args, key, None) is not None}

    def validate_settings(self):
        """Unknown options fail locally; the service checks model-specific compatibility."""
        if set(self.settings.provider_options) - {'reasoning_effort', 'base_url'}:
            raise ValueError('Unsupported OpenAI provider options.')

    def format_request(self, request):
        """Translate the frozen prompt; no vocabulary selection or task rewriting occurs."""
        messages = []
        if request.system_prompt:
            messages.append({'role': 'system', 'content': request.system_prompt})
        messages.append({'role': 'user', 'content': request.user_prompt})
        body = {'model': self.settings.model, 'input': messages,
                'max_output_tokens': self.settings.max_output_tokens, 'store': True}
        effort = self.settings.provider_options.get('reasoning_effort')
        if effort is not None:
            body['reasoning'] = {'effort': effort}
        return body

    def _job(self, native):
        """Keep the native job opaque to general code while mapping lifecycle status."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        write_json(self.workspace / 'batch_job.json', native)
        write_json(self.workspace / f'batch-{native["id"]}.json', native)
        status = native['status']
        status = {'validating': 'queued', 'in_progress': 'running',
                  'finalizing': 'running', 'cancelling': 'running'}.get(status, status)
        if status not in {'queued', 'running', 'completed', 'failed', 'expired', 'cancelled'}:
            status = 'unknown'
        return BatchJob(status, native, native.get('request_counts'))

    def submit_batch(self, attempts, submission_id, on_event):
        """Build/upload OpenAI JSONL; native IDs never become shared state fields."""
        from openai import OpenAIError
        self.workspace.mkdir(parents=True, exist_ok=True)
        batch_attempt_id = attempts[0].attempt_id
        on_event(ProviderEvent('checkpoint', checkpoint={'batch_attempt_id': batch_attempt_id}))
        path = self.workspace / 'batch_input.jsonl'
        # Rebuild deterministically only for an explicitly authorized submission.
        with path.open('w', encoding='utf-8') as output:
            for attempt in attempts:
                entry = {'custom_id': attempt.attempt_id, 'method': 'POST',
                         'url': '/v1/responses', 'body': self.format_request(attempt.request)}
                output.write(json.dumps(entry, ensure_ascii=False) + '\n')
        # Preserve each submission input, even after an explicit uncertain-job retry.
        (self.workspace / f'batch-input-{attempts[0].attempt_id}.jsonl').write_bytes(path.read_bytes())
        try:
            with path.open('rb') as source:
                uploaded = self.client.files.create(file=source, purpose='batch')
            native = self.client.batches.create(input_file_id=uploaded.id,
                endpoint='/v1/responses', completion_window='24h',
                metadata={'submission_id': submission_id, 'batch_attempt_id': batch_attempt_id}).model_dump(mode='json')
            job = self._job(native)
            on_event(ProviderEvent('checkpoint', checkpoint={'job': native, 'batch_attempt_id': batch_attempt_id}))
            return job
        except OpenAIError as error:
            raise ProviderError(str(error)) from error

    def get_batch_status(self, job):
        """Fetch a native job and return the shared envelope."""
        from openai import OpenAIError
        try:
            return self._job(self.client.batches.retrieve(job.handle['id']).model_dump(mode='json'))
        except OpenAIError as error:
            raise ProviderError(str(error)) from error

    def recover_batch(self, submission_id, checkpoint):
        """Reconcile by saved handle, native artifact, or submission metadata."""
        from openai import OpenAIError
        batch_attempt_id = checkpoint.get('batch_attempt_id')
        if not batch_attempt_id:
            return RecoveryReport('safe_to_retry')
        path = self.workspace / 'batch_job.json'
        native = checkpoint.get('job') or (read_json(path) if path.exists() else None)
        if native and (native.get('metadata') or {}).get('batch_attempt_id') != batch_attempt_id:
            native = None
        try:
            if native:
                job = self.get_batch_status(self._job(native))
                return RecoveryReport('finished', job=job)
            for candidate in self.client.batches.list(limit=100):
                if ((candidate.metadata or {}).get('submission_id') == submission_id
                        and (candidate.metadata or {}).get('batch_attempt_id') == batch_attempt_id):
                    return RecoveryReport('finished', job=self._job(candidate.model_dump(mode='json')))
        except (OpenAIError, ProviderError):
            pass
        return RecoveryReport('unknown')

    def collect_batch(self, job, attempts, on_event):
        """Preserve downloaded files and match unordered native records to stable attempts."""
        from openai import OpenAIError
        mapping = {a.attempt_id: a for a in attempts}
        for key in ('output_file_id', 'error_file_id'):
            if not job.handle.get(key):
                continue
            try:
                text = self.client.files.content(job.handle[key]).text
            except OpenAIError as error:
                raise ProviderError(str(error)) from error
            (self.workspace / f'{key}.jsonl').write_text(text, encoding='utf-8')
            (self.workspace / f'file-{job.handle[key]}.jsonl').write_text(text, encoding='utf-8')
            for line in text.splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry['custom_id'] not in mapping:
                    raise ProviderError(f'Unknown batch request ID: {entry["custom_id"]}')
                response = entry.get('response') or {}
                body = response.get('body') or {'status': 'failed', 'error': entry.get('error') or response}
                if entry.get('error'):
                    body['status'] = 'failed'
                yield self.format_result(body, mapping[entry['custom_id']])
