"""Run the common Responses contract against both concrete adapters.

MockTransport exercises real SDK serialization/parsing without network access.
These tests establish client behavior, not compatibility with a live server.
"""
import copy
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from dataclasses import asdict, replace
from pathlib import Path

import httpx2
from openai import DefaultHttpxClient, OpenAI

from ctrl_fin_gen.models import (
    GenerationAttempt,
    GenerationRequest,
    GenerationSettings,
)
from ctrl_fin_gen.providers.openai import OpenAIProvider
from ctrl_fin_gen.providers.vllm_server import VLLMServerProvider
from support import response
from ctrl_fin_gen.providers.base import ProviderError
from ctrl_fin_gen.storage import read_json, read_jsonl, append_jsonl


class ResponsesProviderTests(unittest.TestCase):
    cases = [
        (OpenAIProvider, "openai", {}),
        (VLLMServerProvider, "vllm_server", {"vllm_base_url": "http://unit.test/v1"}),
    ]

    def attempt(self):
        """Use distinctive grouping IDs so accidental replacement is detectable."""
        request = GenerationRequest("r1", "config_7", 3, "Vastaa suomeksi.", "Kirjoita.", {})
        return GenerationAttempt(request, "run_1", "attempt_1")

    def make_provider(self, provider_class, backend, options, handler, mode='immediate'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        client = OpenAI(
            api_key="test-only",
            base_url="http://unit.test/v1",
            max_retries=0,
            http_client=DefaultHttpxClient(
                transport=httpx2.MockTransport(handler)
            ),
        )
        self.addCleanup(client.close)

        settings = GenerationSettings(
            backend=backend,
            model="test-model",
            max_output_tokens=500,
            provider_options=options,
            mode=mode,
        )

        return provider_class(settings, Path(temporary.name), client)

    def test_completed_response(self):
        request = GenerationRequest(
            request_id="request_1",
            configuration_id="config_1",
            repetition_index=1,
            system_prompt="Vastaa suomeksi.",
            user_prompt="Kirjoita koulusta.",
            conditions={},
        )
        attempt = GenerationAttempt(request, "run_1", "attempt_1")

        cases = self.cases

        for provider_class, backend, options in cases:
            with self.subTest(backend=backend):
                received = []

                def handler(http_request):
                    received.append(json.loads(http_request.content))
                    return httpx2.Response(200, json=response())

                provider = self.make_provider(
                    provider_class, backend, options, handler
                )
                events = []

                result = provider.generate(attempt, events.append)

                self.assertEqual(result.status, "completed")
                self.assertEqual(result.text, "Koulussa luettiin kirjaa.")
                self.assertEqual(result.request_id, request.request_id)
                self.assertEqual(result.attempt_id, attempt.attempt_id)
                self.assertEqual(
                    result.configuration_id, request.configuration_id
                )
                self.assertEqual(received[0]["max_output_tokens"], 500)
                self.assertEqual(events[0].kind, "started")

    def test_empty_system_prompt_and_input_immutability(self):
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, lambda request: None)
                request = replace(self.attempt().request, system_prompt='')
                before = asdict(request)
                payload = provider.format_request(request)
                self.assertEqual(payload['input'], [{'role': 'user', 'content': 'Kirjoita.'}])
                self.assertEqual(payload['model'], 'test-model')
                self.assertEqual(asdict(request), before)

    def test_terminal_outcomes_and_all_identity_fields(self):
        for cls, backend, options in self.cases:
            for status in ('completed', 'incomplete', 'failed', 'cancelled'):
                with self.subTest(backend=backend, status=status):
                    native = response(status=status, text='Myös osittainen teksti säilyy.')
                    if status == 'incomplete':
                        native['incomplete_details'] = {'reason': 'max_output_tokens'}
                    if status == 'failed':
                        native['error'] = {'code': 'server_error', 'message': 'Failed'}
                    provider = self.make_provider(cls, backend, options, lambda request: None)
                    attempt = self.attempt()
                    before = copy.deepcopy(native)
                    result = provider.format_result(native, attempt)
                    self.assertEqual(result.status, status)
                    self.assertEqual(result.text, native['output'][0]['content'][0]['text'])
                    self.assertEqual((result.submission_id, result.request_id, result.attempt_id,
                                      result.configuration_id, result.repetition_index),
                                     ('run_1', 'r1', 'attempt_1', 'config_7', 3))
                    self.assertEqual(read_json(provider.workspace / 'attempt_1.json'), before)
                    self.assertEqual(native, before)
                    self.assertEqual(result.native_artifacts, ['provider/attempt_1.json'])
                    if status == 'incomplete':
                        self.assertEqual(result.finish_reason, 'max_output_tokens')

    def test_reasoning_and_tool_items_are_preserved_but_not_educational_text(self):
        native = response(text='Ensimmäinen.')
        native['output'].insert(0, {'type': 'reasoning', 'summary': [{'text': 'Internal reasoning'}]})
        native['output'].append({'type': 'function_call', 'name': 'tool', 'arguments': '{}'})
        native['output'].append({'type': 'message', 'role': 'assistant',
                                 'content': [{'type': 'output_text', 'text': ' Toinen.'}]})
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, lambda request: None)
                result = provider.format_result(native, self.attempt())
                self.assertEqual(result.text, 'Ensimmäinen. Toinen.')
                self.assertEqual(read_json(provider.workspace / 'attempt_1.json'), native)

    def test_refusal_retains_details(self):
        native = response()
        native['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'En voi vastata.'}]
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, lambda request: None)
                result = provider.format_result(native, self.attempt())
                self.assertEqual(result.status, 'refused')
                self.assertEqual(result.text, '')
                self.assertEqual(result.error['message'], 'En voi vastata.')

    def test_missing_usage_is_not_zero_and_missing_output_is_allowed(self):
        for cls, backend, options in self.cases:
            for usage in (None, {}):
                with self.subTest(backend=backend, usage=usage):
                    native = {'status': 'failed', 'output': None, 'usage': usage,
                              'error': {'code': 'failed', 'message': 'No output'}}
                    provider = self.make_provider(cls, backend, options, lambda request: None)
                    result = provider.format_result(native, self.attempt())
                    self.assertEqual(result.text, '')
                    if usage is None:
                        self.assertIsNone(result.usage)
                    else:
                        self.assertIsNone(result.usage['input_tokens'])
                        self.assertIsNone(result.usage['output_tokens'])

    def test_usage_normalization_keeps_native_details_in_artifact(self):
        native = response()
        native['usage'].update(total_tokens=28, output_tokens_details={'reasoning_tokens': 2})
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, lambda request: None)
                result = provider.format_result(native, self.attempt())
                self.assertEqual(result.usage, {'input_tokens': 20, 'output_tokens': 8,
                                              'total_tokens': 28, 'reasoning_tokens': 2})
                self.assertEqual(read_json(provider.workspace / 'attempt_1.json')['usage'], native['usage'])

    def test_local_recovery_needs_no_remote_call(self):
        def forbidden(request):
            self.fail('Local recovery unexpectedly contacted a service')
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, forbidden)
                original = provider.format_result(response(), self.attempt())
                report = provider.recover_request(self.attempt(), {})
                self.assertEqual(report.outcome, 'finished')
                self.assertEqual(report.result, original)

    def test_unknown_without_id_does_not_generate_again(self):
        def forbidden(request):
            self.fail('Unknown recovery must not send a new request')
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, forbidden)
                self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'unknown')

    def test_remote_recovery_waits_then_finishes(self):
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                replies = [response(status='in_progress'), response()]
                calls = []
                def handler(request):
                    calls.append(request.method)
                    self.assertEqual(request.url.path, '/v1/responses/resp_1')
                    return httpx2.Response(200, json=replies.pop(0))
                provider = self.make_provider(cls, backend, options, handler)
                self.assertEqual(provider.recover_request(self.attempt(), {'response_id': 'resp_1'}).outcome, 'running')
                self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'finished')
                self.assertEqual(calls, ['GET', 'GET'])

    def test_unavailable_remote_response_stays_unknown(self):
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                def handler(request):
                    self.assertEqual(request.method, 'GET')
                    return httpx2.Response(404, json={'error': {'message': 'not found'}})
                provider = self.make_provider(cls, backend, options, handler)
                self.assertEqual(provider.recover_request(self.attempt(), {'response_id': 'gone'}).outcome, 'unknown')

    def test_stream_translation_and_terminal_result(self):
        # Calling generate directly tests the inherited protocol. Application-level
        # capability gates are tested separately: vLLM streaming is not enabled yet.
        native = [
            {'type': 'response.created', 'response': response(status='in_progress')},
            {'type': 'response.output_text.delta', 'delta': 'Koulussa '},
            {'type': 'response.output_text.delta', 'delta': 'luettiin kirjaa.'},
            {'type': 'response.completed', 'response': response()},
        ]
        data = ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in native)
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                def handler(request):
                    self.assertTrue(json.loads(request.content)['stream'])
                    return httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'})
                provider = self.make_provider(cls, backend, options, handler, 'stream')
                events = []
                result = provider.generate(self.attempt(), events.append)
                self.assertEqual(result.text, 'Koulussa luettiin kirjaa.')
                self.assertEqual(''.join(e.text for e in events if e.kind == 'text_delta'), result.text)
                self.assertEqual(events[0].kind, 'started')
                self.assertEqual([e.checkpoint['response_id'] for e in events if e.kind == 'checkpoint'], ['resp_1', 'resp_1'])
                self.assertEqual(len(read_jsonl(provider.workspace / 'attempt_1-events.jsonl')), 4)

    def test_broken_stream_keeps_partial_evidence(self):
        native = [
            {'type': 'response.created', 'response': response(status='in_progress')},
            {'type': 'response.output_text.delta', 'delta': 'Kesken jäänyt'},
        ]
        data = ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in native)
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options,
                    lambda request: httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'}), 'stream')
                events = []
                with self.assertRaisesRegex(ProviderError, 'without a final response'):
                    provider.generate(self.attempt(), events.append)
                self.assertEqual(events[-1].text, 'Kesken jäänyt')
                self.assertEqual(read_jsonl(provider.workspace / 'attempt_1-events.jsonl')[-1]['delta'], 'Kesken jäänyt')

    def test_stream_error_event_is_retained(self):
        data = 'event: error\ndata: {"type":"error","message":"stream failed","code":"server_error"}\n\n'
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options,
                    lambda request: httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'}), 'stream')
                with self.assertRaisesRegex(ProviderError, 'stream failed'):
                    provider.generate(self.attempt(), lambda event: None)
                self.assertEqual(read_jsonl(provider.workspace / 'attempt_1-events.jsonl')[0]['message'], 'stream failed')

    def test_rejection_is_terminal_but_connection_loss_is_not(self):
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                calls = []
                def rejected(request):
                    calls.append(request)
                    return httpx2.Response(429, json={'error': {'message': 'quota'}})
                provider = self.make_provider(cls, backend, options, rejected)
                result = provider.generate(self.attempt(), lambda event: None)
                self.assertEqual(result.status, 'failed')
                self.assertEqual(result.error['code'], 'request_rejected')
                self.assertEqual(len(calls), 1)
                def lost(request):
                    raise httpx2.ConnectError('connection lost', request=request)
                provider = self.make_provider(cls, backend, options, lost)
                with self.assertRaises(ProviderError):
                    provider.generate(self.attempt(), lambda event: None)
                self.assertFalse((provider.workspace / 'attempt_1.json').exists())

    def test_injected_clients_remain_callers_responsibility(self):
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, lambda request: None)
                client = provider.client
                provider.close()
                self.assertFalse(client.is_closed())
                self.assertIs(provider.client, client)

    def test_own_client_is_closed_once(self):
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                client = Mock()
                with patch('openai.OpenAI', return_value=client):
                    provider = cls(GenerationSettings(backend, 'm', 100, provider_options=options), directory)
                    self.assertIs(provider.client, client)
                    provider.close()
                    provider.close()
                    client.close.assert_called_once()

    def test_recovered_terminal_file_takes_precedence_over_old_stream_events(self):
        def forbidden(request):
            self.fail('A locally saved terminal response must not be retrieved again')
        for cls, backend, options in self.cases:
            with self.subTest(backend=backend):
                provider = self.make_provider(cls, backend, options, forbidden)
                provider.format_result(response(), self.attempt())
                append_jsonl(provider.workspace / 'attempt_1-events.jsonl', {
                    'type': 'response.created', 'response': response(status='in_progress'),
                })
                self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'finished')
