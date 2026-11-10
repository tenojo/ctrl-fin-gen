"""vLLM configuration and integration tests; all network I/O is simulated."""
import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import httpx2
from openai import OpenAI, DefaultHttpxClient

from ctrl_fin_gen.cli import main
from ctrl_fin_gen.generation import start_generation, resume_generation, collect_results
from ctrl_fin_gen.models import GenerationSettings
from ctrl_fin_gen.providers import PROVIDERS
from ctrl_fin_gen.providers.base import ProviderError
from ctrl_fin_gen.providers.openai import OpenAIProvider
from ctrl_fin_gen.providers.vllm_server import VLLMServerProvider
from ctrl_fin_gen.storage import read_json, read_jsonl
from support import WorkspaceTest, response


class VLLMServerProviderTests(unittest.TestCase):
    def provider(self, options=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        if options is None:
            options = {'vllm_base_url': 'http://localhost:8000/v1'}
        settings = GenerationSettings('vllm_server', 'test-model', 500, provider_options=options)
        return VLLMServerProvider(settings, Path(temporary.name))

    def test_options_from_args(self):
        options = VLLMServerProvider.options_from_args(
            argparse.Namespace(vllm_base_url='http://localhost:8000/v1'))
        self.assertEqual(options, {'vllm_base_url': 'http://localhost:8000/v1'})
        self.assertEqual(VLLMServerProvider.options_from_args(argparse.Namespace()), {})

    def test_cli_flags_coexist_with_openai(self):
        parser = argparse.ArgumentParser()
        OpenAIProvider.add_cli_arguments(parser)
        VLLMServerProvider.add_cli_arguments(parser)
        args = parser.parse_args(['--vllm-base-url', 'http://localhost:8000/v1'])
        self.assertEqual(OpenAIProvider.options_from_args(args), {})
        self.assertEqual(VLLMServerProvider.options_from_args(args),
                         {'vllm_base_url': 'http://localhost:8000/v1'})

    def test_validation_is_local_and_accepts_explicit_http_or_https(self):
        for url in ('http://localhost:8000/v1', 'https://inference.example/v1'):
            with self.subTest(url=url), patch('openai.OpenAI') as constructor:
                provider = self.provider({'vllm_base_url': url})
                provider.validate_settings()
                constructor.assert_not_called()

    def test_missing_or_invalid_url_is_rejected_before_default_host_can_be_used(self):
        options = [{}] + [{'vllm_base_url': value} for value in
                         (None, '', 'localhost:8000', 'http://', 'ftp://host/v1', 42)]
        for value in options:
            with self.subTest(options=value), self.assertRaises(ValueError):
                self.provider(value).validate_settings()

    def test_unknown_options_are_rejected(self):
        with self.assertRaises(ValueError):
            self.provider({'vllm_base_url': 'http://localhost:8000/v1', 'store': True}).validate_settings()

    def test_client_is_lazy_cached_and_not_using_openai_credentials(self):
        client = Mock()
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'unrelated-cloud-key'}, clear=True):
            with patch('openai.OpenAI', return_value=client) as constructor:
                provider = self.provider()
                constructor.assert_not_called()
                self.assertIs(provider.client, client)
                self.assertIs(provider.client, client)
                constructor.assert_called_once_with(
                    base_url='http://localhost:8000/v1', api_key='EMPTY', max_retries=0)
                provider.close()
                provider.close()
                client.close.assert_called_once()

    def test_explicit_server_key_is_used_only_at_runtime(self):
        with patch.dict('os.environ', {'VLLM_API_KEY': 'server-key'}, clear=True):
            with patch('openai.OpenAI') as constructor:
                provider = self.provider()
                provider.client
                self.assertEqual(constructor.call_args.kwargs['api_key'], 'server-key')
                self.assertNotIn('api_key', provider.settings.provider_options)
                provider.close()

    def test_streaming_supported_but_background_batch_is_not(self):
        provider = self.provider()
        self.assertTrue(provider.streaming)
        self.assertFalse(provider.batch_jobs)
        with self.assertRaisesRegex(ProviderError, 'does not support'):
            provider.submit_batch([], 'run_1', lambda event: None)
        with self.assertRaises(ProviderError):
            provider.get_batch_status(None)
        with self.assertRaises(ProviderError):
            provider.collect_batch(None, [], lambda event: None)


# Integration tests use the same real SDK with an in-memory server. Registry
# changes are scoped to the test: registering/enabling the backend remains a
# separate implementation choice, not a side effect of writing this test suite.


class VLLMIntegrationTests(WorkspaceTest):
    def sdk_client(self, handler):
        client = OpenAI(api_key='test-only', base_url='http://vllm.test/v1', max_retries=0,
                        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    def test_shared_runner_streams_and_resumes_without_regenerating(self):
        self.create()
        calls = []
        def handler(request):
            calls.append(request.method)
            self.assertEqual(request.url.host, 'vllm.test')
            native = response()
            data = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in [
                {'type': 'response.output_text.delta', 'delta': 'Koulussa luettiin kirjaa.'},
                {'type': 'response.completed', 'response': native},
            ])
            return httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'})
        settings = GenerationSettings('vllm_server', 'test-model', 100, mode='stream',
                                      provider_options={'vllm_base_url': 'http://vllm.test/v1'})
        folder = self.root / 'experiments/test/generations/stream'
        provider = VLLMServerProvider(settings, folder / 'provider', self.sdk_client(handler))
        text = []
        state = start_generation(self.root, 'test', 'stream', settings, 2, provider, on_text=text.append)
        self.assertEqual(state['status'], 'completed')
        self.assertIn('Koulussa luettiin kirjaa.', ''.join(text))
        self.assertEqual(resume_generation(self.root, 'test', 'stream', provider)['finished_samples'], 2)
        collect_results(self.root, 'test', 'stream', provider)
        self.assertEqual(calls, ['POST', 'POST'])
        self.assertEqual(len(read_jsonl(folder / 'results.jsonl')), 2)

    def test_registered_cli_accepts_vllm_flags_and_preserves_settings(self):
        self.create()
        client = self.sdk_client(lambda request: httpx2.Response(200, json=response()))
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(PROVIDERS, {'vllm_server': VLLMServerProvider}):
            with patch('openai.OpenAI', return_value=client):
                with redirect_stdout(output), redirect_stderr(errors):
                    code = main(['generate', 'test', '--name', 'cli', '--backend', 'vllm_server',
                                 '--model', 'test-model', '--max-length', '100', '--samples', '1',
                                 '--vllm-base-url', 'http://vllm.test/v1'], root=self.root)
        self.assertEqual(code, 0, errors.getvalue())
        self.assertEqual(json.loads(output.getvalue())['status'], 'completed')
        manifest = read_json(self.root / 'experiments/test/generations/cli/manifest.json')
        self.assertEqual(manifest['settings']['provider_options'], {'vllm_base_url': 'http://vllm.test/v1'})

    def test_unsupported_batch_stops_before_files_or_client_creation(self):
        self.create()
        settings = GenerationSettings('vllm_server', 'test-model', 100, mode='batch',
                                      provider_options={'vllm_base_url': 'http://vllm.test/v1'})
        folder = self.root / 'experiments/test/generations/batch'
        with patch('openai.OpenAI') as constructor:
            provider = VLLMServerProvider(settings, folder / 'provider')
            with self.assertRaisesRegex(ValueError, 'background batch'):
                start_generation(self.root, 'test', 'batch', settings, 1, provider)
            constructor.assert_not_called()
        self.assertFalse(folder.exists())
