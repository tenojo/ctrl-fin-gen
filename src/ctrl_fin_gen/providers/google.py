"""Gemini Developer API adapter, authenticated with an API key.

Native request formatting, streaming, Files API I/O, batch status and recovery
live here. Batch rows carry an attempt key: output order is never used, and
identical prompts remain distinguishable. Google dependencies load only when
needed; install ctrl-fin-gen[google] before making service calls.
"""
import hashlib
import json
import math
import os

from .base import Provider, ProviderError
from ..models import BatchJob, GenerationResult, ProviderEvent, RecoveryReport
from ..storage import append_jsonl, read_json, read_jsonl, write_json


class GeminiProvider(Provider):
    """Gemini API generation and background batches, without Cloud project setup."""
    streaming = True
    batch_jobs = True
    OPTIONS = ('api_key_env', 'temperature', 'top_p',
               'top_k', 'seed', 'thinking_budget', 'thinking_level', 'timeout')

    def __init__(self, settings, workspace, client=None):
        """Allow an injected SDK client; production credentials remain lazy."""
        super().__init__(settings, workspace)
        self._client = client
        self._owns_client = client is None

    @classmethod
    def add_cli_arguments(cls, parser):
        """Declare only Google-owned flags; no SDK or credentials are loaded."""
        group = parser.add_argument_group('Google Gemini options')
        group.add_argument('--google-api-key-env', help='Key environment variable; defaults to GOOGLE_API_KEY or GEMINI_API_KEY.')
        group.add_argument('--google-temperature', type=float)
        group.add_argument('--google-top-p', type=float)
        group.add_argument('--google-top-k', type=int)
        group.add_argument('--google-seed', type=int, help='Generation seed, separate from vocabulary sampling.')
        thinking = group.add_mutually_exclusive_group()
        thinking.add_argument('--google-thinking-budget', type=int, help='Model-supported thinking budget; -1 is automatic.')
        thinking.add_argument('--google-thinking-level', choices=['minimal', 'low', 'medium', 'high'],
                              help='Model-supported thinking level, instead of a token budget.')
        group.add_argument('--google-timeout', type=float, help='Request timeout in seconds (default 120).')

    @classmethod
    def options_from_args(cls, args):
        """Keep unspecified values absent so resume detects only explicit overrides."""
        return {key: getattr(args, 'google_' + key) for key in cls.OPTIONS
                if getattr(args, 'google_' + key, None) is not None}

    def validate_settings(self):
        """Validate settings and freeze the key variable name, never its secret value."""
        options = self.settings.provider_options
        if any(key in options for key in ('project', 'location', 'gcs_prefix', 'api')):
            raise ValueError('Cloud project settings are no longer supported; create a fresh Gemini submission.')
        if set(options) - set(self.OPTIONS):
            raise ValueError('Unsupported Gemini provider options.')
        default_env = 'GOOGLE_API_KEY' if os.environ.get('GOOGLE_API_KEY') else 'GEMINI_API_KEY'
        options.setdefault('api_key_env', default_env)
        if not isinstance(options['api_key_env'], str) or not options['api_key_env'].isidentifier():
            raise ValueError('--google-api-key-env must name an environment variable.')
        if 'thinking_level' in options:
            if options['thinking_level'] not in ('minimal', 'low', 'medium', 'high'):
                raise ValueError('Google thinking_level must be minimal, low, medium, or high.')
            if 'thinking_budget' in options:
                raise ValueError('Choose either Google thinking_level or thinking_budget.')
        limits = {'temperature': (0, 2), 'top_p': (0, 1), 'top_k': (1, None),
                  'seed': (0, None), 'thinking_budget': (-1, None), 'timeout': (0.001, None)}
        for key, (low, high) in limits.items():
            if key not in options:
                continue
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Google {key} must be a finite number.')
            if value < low or (high is not None and value > high):
                raise ValueError(f'Google {key} is outside its permitted range.')
            if key in ('top_k', 'seed', 'thinking_budget') and not isinstance(value, int):
                raise ValueError(f'Google {key} must be an integer.')

    @property
    def client(self):
        """Load the key at execution time and explicitly select the Gemini API."""
        if self._client is None:
            self.validate_settings()
            try:
                from google import genai
            except ImportError as error:
                raise ProviderError('Install Google support with: pip install -e ".[google]"') from error
            options = self.settings.provider_options
            api_key = os.environ.get(options['api_key_env'])
            if not api_key or not api_key.strip():
                raise ValueError(f'Set {options["api_key_env"]} to use the Gemini API.')
            self._client = genai.Client(enterprise=False, api_key=api_key,
                http_options={'api_version': 'v1', 'timeout': int(options.get('timeout', 120) * 1000),
                              'retry_options': {'attempts': 1}})
        return self._client

    def format_request(self, request):
        """Map frozen text and generation settings to SDK input; never select vocabulary."""
        options = self.settings.provider_options
        config = {'max_output_tokens': self.settings.max_output_tokens, 'candidate_count': 1,
                  'response_modalities': ['TEXT'], 'automatic_function_calling': {'disable': True},
                  'automatic_continuation': False}
        if request.system_prompt:
            config['system_instruction'] = request.system_prompt
        for key in ('temperature', 'top_p', 'top_k', 'seed'):
            if key in options:
                config[key] = options[key]
        if 'thinking_budget' in options:
            config['thinking_config'] = {'thinking_budget': options['thinking_budget']}
        elif 'thinking_level' in options:
            config['thinking_config'] = {'thinking_level': options['thinking_level'].upper()}
        return {'model': self.settings.model,
                'contents': [{'role': 'user', 'parts': [{'text': request.user_prompt}]}], 'config': config}

    @staticmethod
    def _native(model):
        """Serialize SDK models with REST field names, including unknown/native details."""
        return model.model_dump(mode='json', by_alias=True, exclude_none=True)

    @staticmethod
    def _terminal(body):
        """A transport failure alone is not a terminal response."""
        feedback = body.get('promptFeedback') or {}
        return bool(body.get('error') or feedback.get('blockReason') not in (None, '', 'BLOCK_REASON_UNSPECIFIED')
                    or any(c.get('finishReason') not in (None, '', 'FINISH_REASON_UNSPECIFIED')
                           for c in body.get('candidates') or []))

    def format_result(self, native_output, attempt):
        """Retain native evidence, omit thought parts from text, and normalize outcomes."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / f'{attempt.attempt_id}.json'
        write_json(path, native_output)
        candidates = native_output.get('candidates') or []
        candidate = candidates[0] if candidates else {}
        parts = (candidate.get('content') or {}).get('parts') or []
        text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
        reason = candidate.get('finishReason') or (native_output.get('promptFeedback') or {}).get('blockReason')
        error = native_output.get('error')
        blocked = {'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY'}
        if error:
            status = 'failed'
        elif len(candidates) > 1:
            status, error = 'failed', {'code': 'unexpected_candidates', 'message': 'Expected one candidate; see native output.'}
        elif reason in blocked:
            status, error = 'refused', {'code': 'refused', 'message': reason}
        elif reason in {'MAX_TOKENS', 'CONTINUATION'}:
            status = 'incomplete'
        elif reason == 'STOP' and not any('functionCall' in p for p in parts):
            status = 'completed'
        else:
            status, error = 'failed', {'code': 'unexpected_output', 'message': f'Unexpected finish reason or content: {reason}'}
        native_usage = native_output.get('usageMetadata')
        usage = None
        if native_usage is not None:
            usage = {shared: native_usage.get(native) for shared, native in (
                ('input_tokens', 'promptTokenCount'), ('output_tokens', 'candidatesTokenCount'),
                ('total_tokens', 'totalTokenCount'), ('reasoning_tokens', 'thoughtsTokenCount'))}
        artifacts = [f'provider/{path.name}']
        events = self.workspace / f'{attempt.attempt_id}-events.jsonl'
        if events.exists():
            artifacts.append(f'provider/{events.name}')
        request = attempt.request
        return GenerationResult(
            attempt.submission_id, request.request_id, attempt.attempt_id,
            request.configuration_id, request.repetition_index, status, text=text,
            finish_reason=reason or 'provider_error', model=native_output.get('modelVersion'),
            usage=usage, error=error, native_artifacts=artifacts,
        )

    @staticmethod
    def _merge_chunk(body, chunk):
        """Accumulate streamed parts; the unmodified chunks remain in the journal."""
        candidates = chunk.get('candidates') or []
        if len(candidates) > 1:
            raise ProviderError('Google returned multiple streamed candidates; native evidence is retained.')
        if candidates:
            current = body.setdefault('candidates', [{'content': {'role': 'model', 'parts': []}}])[0]
            current['content']['parts'].extend((candidates[0].get('content') or {}).get('parts') or [])
            for key in ('finishReason', 'finishMessage', 'safetyRatings', 'citationMetadata'):
                if key in candidates[0]:
                    current[key] = candidates[0][key]
        for key in ('usageMetadata', 'modelVersion', 'responseId', 'promptFeedback'):
            if key in chunk:
                body[key] = chunk[key]

    def generate(self, attempt, on_event):
        """Execute once; stream only visible text and preserve all native chunks."""
        client = self.client  # Resolve SDK/auth setup before marking an attempt in flight.
        from google.genai import errors
        import httpx2
        payload = self.format_request(attempt.request)
        self.workspace.mkdir(parents=True, exist_ok=True)
        write_json(self.workspace / f'{attempt.attempt_id}-request.json', payload)
        on_event(ProviderEvent('started', attempt.attempt_id))
        body = {}
        try:
            if self.settings.mode == 'stream':
                body = {}
                stream = client.models.generate_content_stream(**payload)
                try:
                    for model in stream:
                        chunk = self._native(model)
                        append_jsonl(self.workspace / f'{attempt.attempt_id}-events.jsonl', chunk)
                        self._merge_chunk(body, chunk)
                        for candidate in chunk.get('candidates') or []:
                            for part in (candidate.get('content') or {}).get('parts') or []:
                                if part.get('text') and not part.get('thought'):
                                    on_event(ProviderEvent('text_delta', attempt.attempt_id, text=part['text']))
                finally:
                    stream.close()
            else:
                body = self._native(client.models.generate_content(**payload))
            write_json(self.workspace / f'{attempt.attempt_id}.json', body)
            if not self._terminal(body):
                raise ProviderError('Google response ended without a terminal outcome; recovery is required.')
            return self.format_result(body, attempt)
        except errors.APIError as error:
            if not body and error.code in (400, 401, 403, 404, 422, 429):
                return self.format_result({'error': {'code': 'request_rejected', 'message': str(error),
                                                    'details': error.details}}, attempt)
            raise ProviderError(str(error)) from error
        except httpx2.HTTPError as error:
            raise ProviderError(str(error)) from error

    def recover_request(self, attempt, checkpoint):
        """Recover local terminal evidence; Google content responses are not replayed."""
        path = self.workspace / f'{attempt.attempt_id}.json'
        body = read_json(path) if path.exists() else {}
        if not self._terminal(body):
            chunks = read_jsonl(self.workspace / f'{attempt.attempt_id}-events.jsonl')
            if chunks:
                body = {}  # Reconstruct once, rather than duplicating saved partial parts.
            for chunk in chunks:
                self._merge_chunk(body, chunk)
        if self._terminal(body):
            return RecoveryReport('finished', result=self.format_result(body, attempt))
        return RecoveryReport('unknown')

    @staticmethod
    def _batch_token(attempt_id):
        """Build a display-name token from execution identity, not prompt content."""
        return hashlib.sha256(attempt_id.encode()).hexdigest()[:32]

    def _batch_request(self, attempt):
        """Build one REST generation payload for a Gemini batch JSONL row."""
        payload = self.format_request(attempt.request)
        config = payload['config']
        generation = {native: config[key] for key, native in (
            ('max_output_tokens', 'maxOutputTokens'), ('candidate_count', 'candidateCount'),
            ('response_modalities', 'responseModalities'), ('temperature', 'temperature'),
            ('top_p', 'topP'), ('top_k', 'topK'), ('seed', 'seed')) if key in config}
        if 'thinking_config' in config:
            generation['thinkingConfig'] = {native: config['thinking_config'][key] for key, native in (
                ('thinking_budget', 'thinkingBudget'), ('thinking_level', 'thinkingLevel'))
                if key in config['thinking_config']}
        request = {'contents': payload['contents'], 'generationConfig': generation}
        if 'system_instruction' in config:
            request['systemInstruction'] = {'parts': [{'text': config['system_instruction']}]}
        return request

    def _job(self, native):
        """Preserve the full SDK job and expose only the shared lifecycle/opaque handle."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        write_json(self.workspace / 'batch_job.json', native)
        if native.get('name'):
            write_json(self.workspace / f'job-{native["name"].split("/")[-1]}.json', native)
        status = {'JOB_STATE_PENDING': 'queued', 'JOB_STATE_QUEUED': 'queued',
                  'JOB_STATE_RUNNING': 'running', 'JOB_STATE_UPDATING': 'running',
                  'JOB_STATE_CANCELLING': 'running', 'JOB_STATE_SUCCEEDED': 'completed',
                  'JOB_STATE_PARTIALLY_SUCCEEDED': 'completed', 'JOB_STATE_FAILED': 'failed',
                  'JOB_STATE_CANCELLED': 'cancelled', 'JOB_STATE_EXPIRED': 'expired'}.get(native.get('state'), 'unknown')
        return BatchJob(status, native)

    def submit_batch(self, attempts, submission_id, on_event):
        """Upload keyed JSONL with the Gemini Files API; Cloud Storage is not used."""
        client = self.client
        from google.genai import errors
        import httpx2
        token = self._batch_token(attempts[0].attempt_id)
        checkpoint = {'display_name': f'cfg-{submission_id}-{token}'}
        on_event(ProviderEvent('checkpoint', checkpoint=checkpoint))
        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / f'batch-input-{token}.jsonl'
        with path.open('w', encoding='utf-8') as output:
            for attempt in attempts:
                request = self._batch_request(attempt)
                output.write(json.dumps({'key': attempt.attempt_id, 'request': request}, ensure_ascii=False) + '\n')
        try:
            uploaded = client.files.upload(file=str(path), config={
                'mime_type': 'application/jsonl', 'display_name': checkpoint['display_name']})
            write_json(self.workspace / f'batch-file-{token}.json', self._native(uploaded))
            checkpoint = {**checkpoint, 'input_file': uploaded.name}
            on_event(ProviderEvent('checkpoint', checkpoint=checkpoint))
            native = self._native(client.batches.create(model=self.settings.model, src=uploaded.name,
                config={'display_name': checkpoint['display_name']}))
            job = self._job(native)
            on_event(ProviderEvent('checkpoint', checkpoint={**checkpoint, 'job': native}))
            return job
        except (errors.APIError, httpx2.HTTPError) as error:
            raise ProviderError(str(error)) from error

    def get_batch_status(self, job):
        """Query the saved Gemini job without submitting any new work."""
        client = self.client
        from google.genai import errors
        import httpx2
        try:
            return self._job(self._native(client.batches.get(name=job.handle['name'])))
        except (errors.APIError, httpx2.HTTPError) as error:
            raise ProviderError(str(error)) from error

    def recover_batch(self, submission_id, checkpoint):
        """Reconcile by saved job or unique display name, without replaying a submission."""
        if not checkpoint.get('display_name'):
            return RecoveryReport('safe_to_retry')  # The pre-call checkpoint never happened.
        client = self.client
        from google.genai import errors
        import httpx2
        try:
            native = checkpoint.get('job')
            path = self.workspace / 'batch_job.json'
            if native is None and path.exists():
                saved = read_json(path)
                if saved.get('displayName') == checkpoint['display_name']:
                    native = saved
            if native:
                return RecoveryReport('finished', job=self.get_batch_status(self._job(native)))
            for model in client.batches.list(config={'page_size': 100}):
                native = self._native(model)
                if native.get('displayName') == checkpoint['display_name']:
                    return RecoveryReport('finished', job=self._job(native))
        except (errors.APIError, httpx2.HTTPError, ProviderError):
            pass
        return RecoveryReport('unknown')

    def collect_batch(self, job, attempts, on_event):
        """Download keyed Gemini results and retain the complete native JSONL file."""
        self.validate_settings()
        filename = (job.handle.get('dest') or {}).get('fileName')
        if not filename:
            return  # Results may not exist yet; the common runner reports any missing rows.
        client = self.client
        from google.genai import errors
        import httpx2
        self.workspace.mkdir(parents=True, exist_ok=True)
        file_id = hashlib.sha256(filename.encode()).hexdigest()[:24]
        path = self.workspace / f'output-{file_id}.jsonl'
        try:
            path.write_bytes(client.files.download(file=filename))
        except (errors.APIError, httpx2.HTTPError) as error:
            raise ProviderError(str(error)) from error
        mapping = {attempt.attempt_id: attempt for attempt in attempts}
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('key') not in mapping:
                raise ProviderError('Gemini output has an unknown/missing attempt key; native file retained.')
            body = entry.get('response') or {}
            if entry.get('error'):
                body = {**body, 'error': entry['error']}
            if not self._terminal(body):
                raise ProviderError('Gemini batch row lacks a terminal outcome; native file retained.')
            yield self.format_result(body, mapping[entry['key']])

    def close(self):
        """Release owned clients only; injected clients belong to their caller."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
