"""Real SDK against an in-memory HTTP transport; no credentials or paid calls."""
import json
import tempfile
import unittest
import argparse
from unittest.mock import Mock, patch
from pathlib import Path

import httpx2
from openai import DefaultHttpxClient, OpenAI

from ctrl_fin_gen.models import GenerationRequest, GenerationSettings, GenerationAttempt
from ctrl_fin_gen.providers.openai import OpenAIProvider
from ctrl_fin_gen.providers.base import ProviderError
from support import response


class OpenAIProviderTests(unittest.TestCase):
    def provider(self, handler, mode='immediate'):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        client = OpenAI(api_key='test-only', max_retries=0,
                        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
        self.addCleanup(client.close)
        settings = GenerationSettings('openai', 'test-model', 500, mode, {'reasoning_effort': 'low'})
        return OpenAIProvider(settings, Path(temp.name), client)

    def attempt(self):
        request = GenerationRequest('r1', 'c1', 1, 'Vastaa suomeksi.', 'Kirjoita.', {})
        return GenerationAttempt(request, 'submission_1', 'attempt_1')

    def test_payload_and_canonical_result(self):
        received = []
        def handler(request):
            received.append(json.loads(request.content))
            return httpx2.Response(200, json=response())
        provider = self.provider(handler)
        events = []
        result = provider.generate(self.attempt(), events.append)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.text, 'Koulussa luettiin kirjaa.')
        self.assertEqual(received[0]['reasoning'], {'effort': 'low'})
        self.assertEqual(received[0]['max_output_tokens'], 500)
        self.assertEqual(received[0]['input'][0]['role'], 'system')
        self.assertEqual(events[0].kind, 'started')
        self.assertTrue((provider.workspace / 'attempt_1.json').exists())
        self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'finished')




    def test_batch_sdk_roundtrip_and_reconciliation(self):
        job = {'id': 'batch_1', 'status': 'completed', 'metadata': {'submission_id': 'submission_1', 'batch_attempt_id': 'attempt_1'},
               'output_file_id': 'output'}
        def handler(request):
            path = request.url.path
            if path == '/v1/files':
                self.assertIn(b'batch', request.content)
                return httpx2.Response(200, json={'id': 'input', 'object': 'file', 'purpose': 'batch'})
            if path == '/v1/batches' and request.method == 'POST':
                self.assertEqual(json.loads(request.content)['endpoint'], '/v1/responses')
                return httpx2.Response(200, json=job)
            if path == '/v1/batches' and request.method == 'GET':
                return httpx2.Response(200, json={'data': [job], 'object': 'list', 'has_more': False})
            if path == '/v1/batches/batch_1':
                return httpx2.Response(200, json=job)
            if path == '/v1/files/output/content':
                entry = {'custom_id': 'attempt_1', 'response': {'status_code': 200, 'body': response()}}
                return httpx2.Response(200, text=json.dumps(entry) + '\n')
            raise AssertionError(path)
        provider = self.provider(handler, 'batch')
        attempt = self.attempt()
        batch = provider.submit_batch([attempt], 'submission_1', lambda event: None)
        self.assertEqual(batch.status, 'completed')
        self.assertEqual(provider.get_batch_status(batch).status, 'completed')
        results = list(provider.collect_batch(batch, [attempt], lambda event: None))
        self.assertEqual(results[0].request_id, 'r1')
        self.assertEqual(provider.recover_batch('submission_1', {'batch_attempt_id': 'attempt_1'}).job.status, 'completed')
        entry = json.loads((provider.workspace / 'batch_input.jsonl').read_text())
        self.assertEqual(entry['custom_id'], 'attempt_1')
        self.assertNotIn('stream', entry['body'])


    def test_batch_error_rows_and_unknown_ids_retain_native_file(self):
        from ctrl_fin_gen.models import BatchJob
        entry = {'custom_id': 'attempt_1', 'response': None,
                 'error': {'code': 'batch_expired', 'message': 'Expired'}}
        provider = self.provider(lambda request: httpx2.Response(200, text=json.dumps(entry)), 'batch')
        job = BatchJob('expired', {'error_file_id': 'errors'})
        result = list(provider.collect_batch(job, [self.attempt()], lambda event: None))[0]
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.error['code'], 'batch_expired')
        entry['custom_id'] = 'unknown'
        with self.assertRaisesRegex(ProviderError, 'Unknown batch request'):
            list(provider.collect_batch(job, [self.attempt()], lambda event: None))
        self.assertIn('unknown', (provider.workspace / 'error_file_id.jsonl').read_text())

    def test_cli_options_and_validation_are_openai_specific(self):
        parser = argparse.ArgumentParser()
        OpenAIProvider.add_cli_arguments(parser)
        args = parser.parse_args(['--reasoning-effort', 'low', '--base-url', 'https://unit.test/v1'])
        self.assertEqual(OpenAIProvider.options_from_args(args),
                         {'reasoning_effort': 'low', 'base_url': 'https://unit.test/v1'})
        self.assertEqual(OpenAIProvider.options_from_args(parser.parse_args([])), {})
        provider = self.provider(lambda request: None)
        provider.validate_settings()
        provider.settings.provider_options['vllm_base_url'] = 'http://localhost:8000/v1'
        with self.assertRaises(ValueError):
            provider.validate_settings()

    def test_lazy_client_uses_openai_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = GenerationSettings('openai', 'm', 100,
                                          provider_options={'base_url': 'https://cloud.test/v1'})
            client = Mock()
            with patch('openai.OpenAI', return_value=client) as constructor:
                provider = OpenAIProvider(settings, Path(directory))
                constructor.assert_not_called()
                self.assertIs(provider.client, client)
                self.assertIs(provider.client, client)
                constructor.assert_called_once_with(base_url='https://cloud.test/v1', max_retries=0)
                provider.close()
                client.close.assert_called_once()

    def test_batch_reconciliation_uses_submission_attempt_identity(self):
        jobs = [
            {'id': 'wrong_attempt', 'status': 'completed', 'metadata':
             {'submission_id': 'submission_1', 'batch_attempt_id': 'old'}},
            {'id': 'matching', 'status': 'in_progress', 'metadata':
             {'submission_id': 'submission_1', 'batch_attempt_id': 'attempt_1'}},
        ]
        def handler(request):
            self.assertEqual(request.method, 'GET')
            self.assertEqual(request.url.path, '/v1/batches')
            return httpx2.Response(200, json={'data': jobs, 'object': 'list', 'has_more': False})
        provider = self.provider(handler, 'batch')
        report = provider.recover_batch('submission_1', {'batch_attempt_id': 'attempt_1'})
        self.assertEqual(report.job.handle['id'], 'matching')
        self.assertEqual(report.job.status, 'running')
        (provider.workspace / 'batch_job.json').unlink()
        report = provider.recover_batch('submission_1', {'batch_attempt_id': 'missing'})
        self.assertEqual(report.outcome, 'unknown')
        self.assertIsNone(report.job)

    def test_batch_checkpoint_is_saved_before_upload_and_submit(self):
        checkpoints = []
        def handler(request):
            self.assertEqual(checkpoints[0].checkpoint, {'batch_attempt_id': 'attempt_1'})
            if request.url.path == '/v1/files':
                return httpx2.Response(200, json={'id': 'input'})
            return httpx2.Response(200, json={'id': 'batch_1', 'status': 'validating',
                'metadata': {'submission_id': 'submission_1', 'batch_attempt_id': 'attempt_1'}})
        provider = self.provider(handler, 'batch')
        job = provider.submit_batch([self.attempt()], 'submission_1', checkpoints.append)
        self.assertEqual(job.status, 'queued')
        self.assertEqual(checkpoints[-1].checkpoint['job']['id'], 'batch_1')
        self.assertTrue((provider.workspace / 'batch-input-attempt_1.jsonl').exists())
