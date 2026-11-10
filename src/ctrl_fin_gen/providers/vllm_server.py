"""Future VLLMServerProvider: connect to an existing inference server.

Subclass base.Provider, implement native request/result/event translation and
recovery, and register in providers/__init__.py. Server connection options and
CLI declarations belong here. No vLLM service is launched by this client.
"""

import os
from urllib.parse import urlsplit

from .base import Provider, ProviderError
from .responses import ResponsesProvider

TERMINAL = {'completed', 'incomplete', 'failed', 'cancelled'}

class VLLMServerProvider(ResponsesProvider):
    streaming = True
    batch_jobs = False

    @property
    def client(self):
        """Load the optional SDK only when a remote operation is needed."""
        if self._client is None:
            self.validate_settings()
            from openai import OpenAI, OpenAIError
            try:
                self._client = OpenAI(
                    base_url=self.settings.provider_options['vllm_base_url'],
                    api_key=os.environ.get('VLLM_API_KEY') or 'EMPTY',
                    max_retries=0,
                )
            except OpenAIError as error:
                raise ProviderError(str(error)) from error
        return self._client

    @classmethod
    def add_cli_arguments(cls, parser):
        """Keep vLLM-Server-specific flag definitions outside the shared CLI."""
        group = parser.add_argument_group('vLLM Server options')
        group.add_argument('--vllm-base-url', help='vLLM server host address.')

    @classmethod
    def options_from_args(cls, args):
        """Extract only explicitly supplied native options."""
        return {key: getattr(args, key) for key in ('vllm_base_url',)
                if getattr(args, key, None) is not None}

    def validate_settings(self):
        """Reject unsupported native settings before reserving a submission."""
        if set(self.settings.provider_options) - {'vllm_base_url'}:
            raise ValueError('Unsupported vLLM Server provider options.')
        url = self.settings.provider_options.get('vllm_base_url')
        if not isinstance(url, str) or not url or url != url.strip():
            raise ValueError('Supply an explicit --vllm-base-url HTTP(S) address.')
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('--vllm-base-url must be an HTTP(S) server address.')

    def format_request(self, request):
        """Translate one shared request into native input; never called by the runner."""
        messages = []
        if request.system_prompt:
            messages.append({'role': 'system', 'content': request.system_prompt})
        messages.append({'role': 'user', 'content': request.user_prompt})
        body = {'model': self.settings.model, 'input': messages,
                'max_output_tokens': self.settings.max_output_tokens}
        return body
