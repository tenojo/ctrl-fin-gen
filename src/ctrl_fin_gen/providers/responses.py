import json

from .base import Provider, ProviderError
from ..models import BatchJob, GenerationResult, ProviderEvent, RecoveryReport
from ..storage import append_jsonl, read_json, read_jsonl, write_json

TERMINAL = {'completed', 'incomplete', 'failed', 'cancelled'}

class ResponsesProvider(Provider):
    def __init__(self, settings, workspace, client=None):
        """Accept an optional injected SDK client, without loading a production client."""
        super().__init__(settings, workspace)
        self._client = client
        self._owns_client = client is None

    def format_result(self, native_output, attempt):
            """Preserve the complete native body before extracting one terminal outcome."""
            self.workspace.mkdir(parents=True, exist_ok=True)
            path = self.workspace / f'{attempt.attempt_id}.json'
            write_json(path, native_output)
            parts, refusals = [], []
            for item in native_output.get('output') or []:
                if item.get('type') != 'message' or item.get('role', 'assistant') != 'assistant':
                    continue
                for part in item.get('content') or []:
                    if part.get('type') == 'output_text':
                        parts.append(part['text'])
                    elif part.get('type') == 'refusal':
                        refusals.append(part['refusal'])
            details = native_output.get('incomplete_details') or {}
            error = native_output.get('error')
            if refusals:
                error = {'code': 'refused', 'message': '\n'.join(refusals)}
            request = attempt.request
            native_usage = native_output.get('usage')
            usage = None
            if native_usage is not None:
                usage = {key: native_usage.get(key) for key in ('input_tokens', 'output_tokens', 'total_tokens')}
                usage['reasoning_tokens'] = (native_usage.get('output_tokens_details') or {}).get('reasoning_tokens')
            return GenerationResult(
                attempt.submission_id, request.request_id, attempt.attempt_id,
                request.configuration_id, request.repetition_index,
                'refused' if refusals else native_output.get('status', 'failed'),
                text=''.join(parts), model=native_output.get('model'),
                usage=usage, error=error,
                finish_reason=details.get('reason') or ('provider_error' if error or native_output.get('status') == 'failed' else 'stop'),
                native_artifacts=[f'provider/{path.name}'],
            )
    
    def generate(self, attempt, on_event):
        """Journal native stream evidence and emit only shared events to the runner."""
        from openai import OpenAIError
        self.workspace.mkdir(parents=True, exist_ok=True)
        body = self.format_request(attempt.request)
        body['metadata'] = {'submission_id': attempt.submission_id,
                            'request_id': attempt.request.request_id}
        write_json(self.workspace / f'{attempt.attempt_id}-request.json', body)
        client = self.client  # Credential/setup failures happen before marking work in-flight.
        on_event(ProviderEvent('started', attempt.attempt_id))
        try:
            if self.settings.mode == 'stream':
                final = None
                with client.responses.create(**body, stream=True) as stream:
                    for event in stream:
                        data = event.model_dump(mode='json')
                        append_jsonl(self.workspace / f'{attempt.attempt_id}-events.jsonl', data)
                        if 'response' in data:
                            response = data['response']
                            write_json(self.workspace / f'{attempt.attempt_id}.json', response)
                            on_event(ProviderEvent('checkpoint', attempt.attempt_id,
                                                    checkpoint={'response_id': response.get('id')}))
                            if response.get('status') in TERMINAL:
                                final = response
                        if data['type'] == 'response.output_text.delta':
                            on_event(ProviderEvent('text_delta', attempt.attempt_id, text=data['delta']))
                        if data['type'] == 'error':
                            raise ProviderError(data.get('message', 'Stream error.'))
                if final is None:
                    raise ProviderError('Stream ended without a final response.')
            else:
                final = client.responses.create(**body).model_dump(mode='json')
            if final.get('status') not in TERMINAL:
                raise ProviderError('Response is not terminal; recovery is required.')
            return self.format_result(final, attempt)
        except OpenAIError as error:
            # Clear rejections are known finished failures, not ambiguous lost responses.
            if getattr(error, 'status_code', None) in (400, 401, 403, 404, 422, 429):
                return self.format_result({'status': 'failed', 'error': {
                    'code': 'request_rejected', 'message': str(error),
                    'details': getattr(error, 'body', None),
                }}, attempt)
            raise ProviderError(str(error)) from error
        

    def recover_request(self, attempt, checkpoint):
        """Use native saved evidence first, then retrieve a known stored response."""
        from openai import OpenAIError
        path = self.workspace / f'{attempt.attempt_id}.json'
        body = read_json(path) if path.exists() else None
        # A recovered final response may be newer than the interrupted stream
        # journal. Never replace terminal evidence with an older progress event.
        if body and body.get('status') in TERMINAL:
            return RecoveryReport('finished', result=self.format_result(body, attempt))
        for event in read_jsonl(self.workspace / f'{attempt.attempt_id}-events.jsonl'):
            if 'response' in event:
                body = event['response']
        if body and body.get('status') in TERMINAL:
            return RecoveryReport('finished', result=self.format_result(body, attempt))
        response_id = (body or {}).get('id') or checkpoint.get('response_id')
        if not response_id:
            return RecoveryReport('unknown')
        try:
            body = self.client.responses.retrieve(response_id).model_dump(mode='json')
        except OpenAIError:
            return RecoveryReport('unknown')
        write_json(path, body)
        if body.get('status') in TERMINAL:
            return RecoveryReport('finished', result=self.format_result(body, attempt))
        return RecoveryReport('running')
    
    def close(self):
        """Close only clients created by this provider; injected clients belong to callers."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
