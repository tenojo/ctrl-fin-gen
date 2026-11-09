"""Exercise the real SDK against an in-memory HTTP transport."""

import json
import tempfile
import unittest
from pathlib import Path

import httpx2
from openai import DefaultHttpxClient, OpenAI

from ctrl_fin_gen.providers.openai import OpenAIProvider
from support import response


class OpenAIProviderTests(unittest.TestCase):
    def provider(self, handler):
        client = OpenAI(
            api_key='test-key-never-sent-to-a-network', max_retries=0,
            http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )
        self.addCleanup(client.close)
        return OpenAIProvider(client)

    def test_sdk_responses_payload_and_terminal_event(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx2.Response(200, json=response())

        provider = self.provider(handler)
        events = []
        body = {'model': 'test-model', 'input': [{'role': 'user', 'content': 'Kirjoita suomeksi.'}],
                'max_output_tokens': 500, 'reasoning': {'effort': 'low'}, 'store': True}
        result = provider.generate(body, False, events.append)
        self.assertEqual(requests[0].url.path, '/v1/responses')
        self.assertEqual(json.loads(requests[0].content), body)
        self.assertEqual(result['id'], 'resp_1')
        self.assertEqual(events[-1]['type'], 'response.completed')

    def test_sdk_parses_sse_events_and_returns_final_body(self):
        events = [
            {'type': 'response.created', 'response': response(status='in_progress')},
            {'type': 'response.output_text.delta', 'delta': 'Koulu on tuttu.'},
            {'type': 'response.completed', 'response': response(text='Koulu on tuttu.')},
        ]
        data = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in events)

        def handler(request):
            self.assertTrue(json.loads(request.content)['stream'])
            return httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'})

        received = []
        result = self.provider(handler).generate({'model': 'test-model', 'input': 'Kirjoita.'}, True, received.append)
        self.assertEqual([event['type'] for event in received], [event['type'] for event in events])
        self.assertEqual(result['output'][0]['content'][0]['text'], 'Koulu on tuttu.')

    def test_sdk_rejects_a_stream_without_a_terminal_response(self):
        data = 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Kesken"}\n\n'
        provider = self.provider(lambda request: httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'}))
        with self.assertRaisesRegex(RuntimeError, 'without a final response'):
            provider.generate({'model': 'test-model', 'input': 'Kirjoita.'}, True, lambda event: None)

    def test_sdk_batch_upload_submit_status_list_and_download(self):
        calls = []
        job = {'id': 'batch_1', 'status': 'completed', 'metadata': {'submission_id': 'submission_1'},
               'output_file_id': 'file_output'}

        def handler(request):
            calls.append((request.method, request.url.path))
            if request.url.path == '/v1/files':
                self.assertIn(b'batch', request.content)
                return httpx2.Response(200, json={'id': 'file_input', 'object': 'file', 'purpose': 'batch'})
            if request.url.path == '/v1/batches' and request.method == 'POST':
                self.assertEqual(json.loads(request.content), {
                    'input_file_id': 'file_input', 'endpoint': '/v1/responses',
                    'completion_window': '24h', 'metadata': {'submission_id': 'submission_1'},
                })
                return httpx2.Response(200, json=job)
            if request.url.path == '/v1/batches' and request.method == 'GET':
                return httpx2.Response(200, json={'object': 'list', 'data': [job], 'has_more': False})
            if request.url.path == '/v1/batches/batch_1':
                return httpx2.Response(200, json=job)
            if request.url.path == '/v1/files/file_output/content':
                return httpx2.Response(200, text='{"teksti": "Järvi jäätyi."}\n')
            raise AssertionError(f'Unexpected endpoint: {request.url.path}')

        provider = self.provider(handler)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.jsonl'
            path.write_text('{"custom_id": "test"}\n')
            self.assertEqual(provider.submit_batch(path, 'submission_1')['id'], 'batch_1')
        self.assertEqual(provider.find_batch('submission_1')['id'], 'batch_1')
        self.assertEqual(provider.retrieve_batch('batch_1')['status'], 'completed')
        self.assertIn('Järvi jäätyi.', provider.download('file_output'))
        self.assertEqual(len(calls), 5)
