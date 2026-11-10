"""Gemini SDK wire tests: no real credentials, cloud requests, or storage.

Install .[google] to run these tests. Generation, resumable file upload, batch
jobs and output download all go through the real SDK and MockTransport.
"""
import argparse
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import httpx2
    from google import genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from ctrl_fin_gen.cli import main, parse_args
from ctrl_fin_gen.generation import start_generation, resume_generation, collect_results, get_status
from ctrl_fin_gen.models import GenerationSettings, GenerationRequest, GenerationAttempt
from ctrl_fin_gen.providers import get_provider_class
from ctrl_fin_gen.providers.google import GeminiProvider
from ctrl_fin_gen.providers.base import ProviderError
from ctrl_fin_gen.providers.openai import OpenAIProvider
from ctrl_fin_gen.storage import read_json, read_jsonl
from support import WorkspaceTest


def google_response(text='Koulussa luettiin kirjaa.', reason='STOP'):
    return {'candidates': [{'index': 0, 'content': {'role': 'model', 'parts': [{'text': text}]},
                            'finishReason': reason}], 'modelVersion': 'test-model-version',
            'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 6,
                              'totalTokenCount': 20, 'thoughtsTokenCount': 2}}


@unittest.skipUnless(GOOGLE_AVAILABLE, 'Install .[google] to test the Google SDK adapter')
class GoogleProviderTests(WorkspaceTest):
    def sdk(self, handler):
        client = genai.Client(enterprise=False, api_key='test-only',
            http_options={'api_version': 'v1', 'retry_options': {'attempts': 1},
                'httpx_client': httpx2.Client(transport=httpx2.MockTransport(handler))})
        self.addCleanup(client.close)
        return client

    def settings(self, mode='immediate', **options):
        return GenerationSettings('gemini', 'test-model', 500, mode, options)

    def attempt(self, attempt_id='attempt_1', request_id='r1'):
        return GenerationAttempt(GenerationRequest(request_id, 'config_7', 3,
            'Vastaa suomeksi.', 'Kirjoita koulusta.', {}), 'submission_1', attempt_id)

    def provider(self, handler=None, mode='immediate', **options):
        handler = handler or (lambda request: httpx2.Response(200, json=google_response()))
        directory = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(directory.cleanup)
        return GeminiProvider(self.settings(mode, **options), Path(directory.name), self.sdk(handler))

    def test_lazy_client_and_cli_flags(self):
        parser = argparse.ArgumentParser()
        OpenAIProvider.add_cli_arguments(parser)
        GeminiProvider.add_cli_arguments(parser)
        args = parser.parse_args(['--google-api-key-env', 'PILOT_KEY', '--google-temperature', '0',
                                  '--google-thinking-budget', '0'])
        self.assertEqual(GeminiProvider.options_from_args(args),
                         {'api_key_env': 'PILOT_KEY', 'temperature': 0.0, 'thinking_budget': 0})
        self.assertEqual(GeminiProvider.options_from_args(parser.parse_args([])), {})
        env = {'GEMINI_API_KEY': 'test-secret', 'GOOGLE_CLOUD_PROJECT': 'ignored-project',
               'GOOGLE_GENAI_USE_ENTERPRISE': 'true'}
        with patch.dict('os.environ', env, clear=True), patch('google.genai.Client', return_value=Mock()) as factory:
            provider = GeminiProvider(self.settings(), self.root)
            provider.validate_settings()
            self.assertEqual(provider.settings.provider_options, {'api_key_env': 'GEMINI_API_KEY'})
            factory.assert_not_called()
            provider.client
            factory.assert_called_once_with(enterprise=False, api_key='test-secret',
                http_options={'api_version': 'v1', 'timeout': 120000, 'retry_options': {'attempts': 1}})
            provider.close()

    def test_key_environment_precedence_custom_variable_and_resume_pin(self):
        env = {'GOOGLE_API_KEY': 'google-secret', 'GEMINI_API_KEY': 'gemini-secret', 'PILOT_KEY': 'pilot-secret'}
        with patch.dict('os.environ', env, clear=True), patch('google.genai.Client', return_value=Mock()) as factory:
            provider = GeminiProvider(self.settings(), self.root)
            provider.client
            self.assertEqual(factory.call_args.kwargs['api_key'], 'google-secret')
            provider.close()
            custom = GeminiProvider(self.settings(api_key_env='PILOT_KEY'), self.root)
            custom.client
            self.assertEqual(factory.call_args.kwargs['api_key'], 'pilot-secret')
            custom.close()
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'gemini-secret'}, clear=True):
            provider = GeminiProvider(self.settings(), self.root)
            provider.validate_settings()
            with patch.dict('os.environ', env):
                provider.validate_settings()
                self.assertEqual(provider.settings.provider_options['api_key_env'], 'GEMINI_API_KEY')

    def test_missing_key_and_invalid_options_fail_clearly(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Set GEMINI_API_KEY'):
                GeminiProvider(self.settings(), self.root).client
        cases = [{'api_key_env': 'not a variable'}, {'project': 'old-cloud-project'},
                 {'gcs_prefix': 'gs://bucket'}, {'temperature': -1}, {'top_p': 2},
                 {'top_k': 0}, {'seed': True}, {'thinking_budget': -2},
                 {'thinking_level': 'invalid'}, {'thinking_level': 'low', 'thinking_budget': 100},
                 {'timeout': float('inf')}, {'n': 2}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.provider(**options).validate_settings()

    def test_real_sdk_payload_and_canonical_result(self):
        received = []
        def handler(request):
            self.assertEqual(request.url.host, 'generativelanguage.googleapis.com')
            self.assertEqual(request.headers['x-goog-api-key'], 'test-only')
            self.assertNotIn('authorization', request.headers)
            self.assertTrue(request.url.path.endswith('/models/test-model:generateContent'))
            received.append(json.loads(request.content))
            return httpx2.Response(200, json=google_response())
        provider = self.provider(handler, temperature=0, thinking_budget=0)
        attempt, events = self.attempt(), []
        before = copy.deepcopy(attempt.request)
        result = provider.generate(attempt, events.append)
        wire = received[0]
        self.assertEqual(wire['contents'][0]['parts'][0]['text'], attempt.request.user_prompt)
        self.assertEqual(wire['systemInstruction']['parts'][0]['text'], attempt.request.system_prompt)
        self.assertEqual(wire['generationConfig']['maxOutputTokens'], 500)
        self.assertEqual(wire['generationConfig']['candidateCount'], 1)
        thinking = wire['generationConfig']['thinkingConfig']
        self.assertEqual(thinking.get('thinkingBudget', thinking.get('thinking_budget')), 0)
        self.assertEqual((result.request_id, result.attempt_id, result.configuration_id, result.repetition_index),
                         ('r1', 'attempt_1', 'config_7', 3))
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.text, 'Koulussa luettiin kirjaa.')
        self.assertEqual(result.usage, {'input_tokens': 12, 'output_tokens': 6,
                                      'total_tokens': 20, 'reasoning_tokens': 2})
        self.assertEqual(events[0].kind, 'started')
        self.assertEqual(attempt.request, before)
        saved = read_json(provider.workspace / 'attempt_1.json')
        saved.pop('sdkHttpResponse', None)  # SDK-added response headers are retained too.
        self.assertEqual(saved, google_response())

    def test_thinking_level_reaches_sdk_and_batch_formats(self):
        received = []
        def handler(request):
            received.append(json.loads(request.content))
            return httpx2.Response(200, json=google_response())
        provider = self.provider(handler, thinking_level='low')
        provider.generate(self.attempt(), lambda event: None)
        thinking = received[0]['generationConfig']['thinkingConfig']
        self.assertEqual(thinking.get('thinkingLevel', thinking.get('thinking_level')), 'LOW')
        self.assertEqual(provider._batch_request(self.attempt())['generationConfig']['thinkingConfig'],
                         {'thinkingLevel': 'LOW'})

    def test_empty_system_prompt_and_unknown_usage(self):
        provider = self.provider()
        request = replace(self.attempt().request, system_prompt='')
        self.assertNotIn('system_instruction', provider.format_request(request)['config'])
        native = google_response()
        native.pop('usageMetadata')
        self.assertIsNone(provider.format_result(native, self.attempt()).usage)

    def test_terminal_outcomes_and_thoughts_remain_in_native_artifacts(self):
        provider = self.provider()
        cases = [('STOP', 'completed'), ('MAX_TOKENS', 'incomplete'), ('CONTINUATION', 'incomplete'),
                 ('SAFETY', 'refused'), ('RECITATION', 'refused'), ('OTHER', 'failed')]
        for reason, expected in cases:
            with self.subTest(reason=reason):
                native = google_response(reason=reason)
                native['candidates'][0]['content']['parts'].insert(0, {'text': 'Internal thought', 'thought': True})
                result = provider.format_result(native, self.attempt())
                self.assertEqual(result.status, expected)
                self.assertNotIn('Internal thought', result.text)
                self.assertEqual(read_json(provider.workspace / 'attempt_1.json'), native)
        result = provider.format_result({'promptFeedback': {'blockReason': 'SAFETY'}}, self.attempt())
        self.assertEqual((result.status, result.text), ('refused', ''))

    def test_no_silent_candidate_or_tool_output_selection(self):
        provider = self.provider()
        native = google_response()
        native['candidates'].append(copy.deepcopy(native['candidates'][0]))
        self.assertEqual(provider.format_result(native, self.attempt()).error['code'], 'unexpected_candidates')
        native = google_response()
        native['candidates'][0]['content']['parts'] = [{'functionCall': {'name': 'unexpected', 'args': {}}}]
        self.assertEqual(provider.format_result(native, self.attempt()).status, 'failed')

    def test_streaming_assembly_preserves_usage_only_final_chunk(self):
        chunks = [
            {'candidates': [{'content': {'parts': [{'text': 'Thinking', 'thought': True}]}}]},
            {'candidates': [{'content': {'parts': [{'text': 'Koulussa '}]}}]},
            {'candidates': [{'content': {'parts': [{'text': 'luettiin.'}]}, 'finishReason': 'STOP'}]},
            {'usageMetadata': {'promptTokenCount': 4, 'candidatesTokenCount': 2}},
        ]
        data = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks)
        def handler(request):
            self.assertTrue(request.url.path.endswith(':streamGenerateContent'))
            return httpx2.Response(200, text=data, headers={'content-type': 'text/event-stream'})
        provider = self.provider(handler, 'stream')
        events = []
        result = provider.generate(self.attempt(), events.append)
        self.assertEqual(result.text, 'Koulussa luettiin.')
        self.assertEqual(''.join(e.text for e in events if e.kind == 'text_delta'), result.text)
        self.assertEqual(result.usage['input_tokens'], 4)
        self.assertIsNone(result.usage['reasoning_tokens'])
        self.assertEqual(len(read_jsonl(provider.workspace / 'attempt_1-events.jsonl')), 4)
        self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'finished')

    def test_broken_stream_keeps_text_but_does_not_fabricate_terminal_outcome(self):
        data = 'data: ' + json.dumps({'candidates': [{'content': {'parts': [{'text': 'Kesken'}]}}]}) + '\n\n'
        provider = self.provider(lambda request: httpx2.Response(200, text=data,
                                 headers={'content-type': 'text/event-stream'}), 'stream')
        with self.assertRaisesRegex(ProviderError, 'terminal outcome'):
            provider.generate(self.attempt(), lambda event: None)
        self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'unknown')
        # Repeated recovery must not duplicate the partial text.
        self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'unknown')
        self.assertEqual(len(read_jsonl(provider.workspace / 'attempt_1-events.jsonl')), 1)

    def test_local_recovery_and_terminal_journal_need_no_service(self):
        def forbidden(request):
            self.fail('Local recovery made a service request')
        provider = self.provider(forbidden)
        original = provider.format_result(google_response(), self.attempt())
        self.assertEqual(provider.recover_request(self.attempt(), {}).result, original)
        (provider.workspace / 'attempt_1.json').unlink()
        from ctrl_fin_gen.storage import append_jsonl
        append_jsonl(provider.workspace / 'attempt_1-events.jsonl', google_response())
        self.assertEqual(provider.recover_request(self.attempt(), {}).result.text, original.text)
        self.assertEqual(provider.recover_request(self.attempt('missing'), {}).outcome, 'unknown')

    def test_api_rejection_vs_unknown_connection_and_no_sdk_retries(self):
        for status in (400, 429, 500):
            with self.subTest(status=status):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx2.Response(status, json={'error': {'code': status, 'message': 'test error'}})
                provider = self.provider(handler)
                if status == 500:
                    with self.assertRaises(ProviderError):
                        provider.generate(self.attempt(), lambda event: None)
                else:
                    self.assertEqual(provider.generate(self.attempt(), lambda event: None).status, 'failed')
                self.assertEqual(len(calls), 1)
        def disconnected(request):
            raise httpx2.ConnectError('lost', request=request)
        provider = self.provider(disconnected)
        with self.assertRaises(ProviderError):
            provider.generate(self.attempt(), lambda event: None)
        self.assertEqual(provider.recover_request(self.attempt(), {}).outcome, 'unknown')

    def test_sdk_automatic_continuation_is_disabled(self):
        calls = []
        def handler(request):
            calls.append(request)
            native = google_response(reason='CONTINUATION')
            native['continuationToken'] = 'next-part'
            return httpx2.Response(200, json=native)
        provider = self.provider(handler)
        self.assertEqual(provider.generate(self.attempt(), lambda event: None).status, 'incomplete')
        self.assertEqual(len(calls), 1)

    def test_injected_clients_are_not_closed(self):
        client = Mock()
        provider = GeminiProvider(self.settings(), self.root, client)
        provider.close()
        client.close.assert_not_called()

    def batch_provider(self):
        """Simulate the complete Files API upload/download and native batch endpoints."""
        state = {'native': None, 'creates': 0, 'uploads': 0, 'listing': [],
                 'checkpoints': [], 'rows': [], 'output': b'', 'fail_create': False}
        def handler(request):
            self.assertEqual(request.url.host, 'generativelanguage.googleapis.com')
            if request.url.path == '/upload/v1/files':
                self.assertTrue(state['checkpoints'])
                self.assertEqual(request.headers['x-goog-upload-header-content-type'], 'application/jsonl')
                return httpx2.Response(200, json={}, headers={
                    'x-goog-upload-url': 'https://generativelanguage.googleapis.com/upload/test'})
            if request.url.path == '/upload/test':
                state['uploads'] += 1
                state['rows'] = [json.loads(row) for row in request.content.decode().splitlines()]
                return httpx2.Response(200, json={'file': {'name': 'files/input'}},
                                       headers={'x-goog-upload-status': 'final'})
            if request.method == 'POST':
                state['creates'] += 1
                self.assertTrue(request.url.path.endswith('/models/test-model:batchGenerateContent'))
                body = json.loads(request.content)['batch']
                self.assertEqual(body['inputConfig']['fileName'], 'files/input')
                state['native'] = {'name': 'batches/123', 'metadata': {
                    'displayName': body['displayName'], 'state': 'BATCH_STATE_PENDING'}}
                if state['fail_create']:
                    raise httpx2.ConnectError('lost after submission', request=request)
                return httpx2.Response(200, json=state['native'])
            if request.url.path.endswith('/files/output:download'):
                return httpx2.Response(200, content=state['output'])
            if request.url.path.endswith('/batches'):
                self.assertNotIn('filter', request.url.params)
                return httpx2.Response(200, json={'operations': state['listing']})
            self.assertTrue(request.url.path.endswith('/batches/123'))
            return httpx2.Response(200, json=state['native'])
        provider = self.provider(handler, 'batch')
        return provider, state

    def finish_batch(self, state, rows):
        state['native']['metadata'].update(state='BATCH_STATE_SUCCEEDED', output={'responsesFile': 'files/output'})
        state['output'] = ('\n'.join(json.dumps(row) for row in rows) + '\n').encode('utf-8')

    def test_batch_unordered_identical_prompts_and_error_rows(self):
        provider, state = self.batch_provider()
        attempts = [self.attempt(), self.attempt('attempt_2', 'r2')]
        job = provider.submit_batch(attempts, 'submission_1', state['checkpoints'].append)
        self.assertEqual(job.status, 'queued')
        rows = state['rows']
        self.assertEqual([row['key'] for row in rows], ['attempt_1', 'attempt_2'])
        self.assertEqual(rows[0]['request'], rows[1]['request'])
        self.assertEqual(rows[0]['request']['systemInstruction']['parts'][0]['text'], 'Vastaa suomeksi.')
        self.finish_batch(state, [
            {'key': rows[1]['key'], 'response': google_response('Toinen.')},
            {'key': rows[0]['key'], 'error': {'code': 400, 'message': 'Bad request'}},
        ])
        job = provider.get_batch_status(job)
        self.assertEqual(job.status, 'completed')
        self.assertEqual(job.handle['dest']['fileName'], 'files/output')
        results = list(provider.collect_batch(job, attempts, lambda event: None))
        self.assertEqual([(r.request_id, r.status) for r in results], [('r2', 'completed'), ('r1', 'failed')])
        self.assertEqual(results[1].error['message'], 'Bad request')
        self.assertEqual(results, list(provider.collect_batch(job, attempts, lambda event: None)))
        self.assertTrue(list(provider.workspace.glob('output-*.jsonl')))

    def test_batch_unknown_key_and_nonterminal_rows_preserve_native_file(self):
        for row in ({'key': 'unknown', 'response': google_response()},
                    {'key': 'attempt_1', 'response': {}}):
            with self.subTest(row=row):
                provider, state = self.batch_provider()
                job = provider.submit_batch([self.attempt()], 'submission_1', state['checkpoints'].append)
                self.finish_batch(state, [row])
                job = provider.get_batch_status(job)
                with self.assertRaises(ProviderError):
                    list(provider.collect_batch(job, [self.attempt()], lambda event: None))
                self.assertEqual(next(provider.workspace.glob('output-*.jsonl')).read_bytes(), state['output'])

    def test_batch_reconciliation_matches_exact_display_name_without_resubmitting(self):
        provider, state = self.batch_provider()
        provider.submit_batch([self.attempt()], 'submission_1', state['checkpoints'].append)
        checkpoint = state['checkpoints'][-1].checkpoint
        self.assertEqual(provider.recover_batch('submission_1', checkpoint).job.status, 'queued')
        checkpoint = state['checkpoints'][0].checkpoint
        (provider.workspace / 'batch_job.json').unlink()
        other = {'name': 'batches/456', 'metadata': {'displayName': 'not-ours', 'state': 'BATCH_STATE_RUNNING'}}
        state['listing'] = [other, state['native']]
        self.assertEqual(provider.recover_batch('submission_1', checkpoint).job.handle['name'], 'batches/123')
        (provider.workspace / 'batch_job.json').unlink()
        state['listing'] = [other]
        self.assertEqual(provider.recover_batch('submission_1', checkpoint).outcome, 'unknown')
        self.assertEqual(provider.recover_batch('submission_1', {}).outcome, 'safe_to_retry')
        self.assertEqual(state['creates'], 1)

    def test_batch_unknown_submission_can_be_reconciled(self):
        provider, state = self.batch_provider()
        state['fail_create'] = True
        with self.assertRaises(ProviderError):
            provider.submit_batch([self.attempt()], 'submission_1', state['checkpoints'].append)
        self.assertEqual(state['creates'], 1)
        state['listing'] = [state['native']]
        report = provider.recover_batch('submission_1', state['checkpoints'][-1].checkpoint)
        self.assertEqual(report.job.handle['name'], 'batches/123')
        self.assertEqual(state['uploads'], 1)
        with patch.object(provider.client.batches, 'get', side_effect=httpx2.ConnectError('lost')):
            with self.assertRaises(ProviderError):
                provider.get_batch_status(report.job)
            self.assertEqual(provider.recover_batch('submission_1',
                state['checkpoints'][-1].checkpoint).outcome, 'unknown')

    def test_shared_runner_batch_collection_resume_and_missing_results(self):
        self.create()
        provider, state = self.batch_provider()
        original_submit = provider.submit_batch
        def submit(attempts, submission_id, callback):
            def checkpoint(event):
                state['checkpoints'].append(event)
                callback(event)
            return original_submit(attempts, submission_id, checkpoint)
        provider.submit_batch = submit
        run = start_generation(self.root, 'test', 'gemini', provider.settings, 2, provider)
        self.assertEqual(run['status'], 'queued')
        self.assertEqual(get_status(self.root, 'test', 'gemini', provider)['provider_status'], 'queued')
        rows = [{'key': row['key'], 'response': google_response()} for row in reversed(state['rows'])]
        self.finish_batch(state, rows[:1])
        partial = collect_results(self.root, 'test', 'gemini', provider)
        self.assertEqual((partial['status'], partial['remaining_samples']), ('missing_results', 1))
        self.finish_batch(state, rows)
        collected = collect_results(self.root, 'test', 'gemini', provider)
        self.assertEqual(collected['finished_samples'], 2)
        resume_generation(self.root, 'test', 'gemini', provider)
        self.assertEqual(state['creates'], 1)

    def test_cli_alias_key_not_saved_and_resume_override_guard(self):
        self.create()
        self.assertIs(get_provider_class('google_cloud'), get_provider_class('gemini'))
        client = self.sdk(lambda request: httpx2.Response(200, json=google_response()))
        output, error = io.StringIO(), io.StringIO()
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'unique-secret-for-test'}, clear=True):
            with patch('google.genai.Client', return_value=client):
                with redirect_stdout(output), redirect_stderr(error):
                    code = main(['generate', 'test', '--name', 'gemini-cli', '--backend', 'gemini',
                        '--model', 'test-model', '--max-length', '100', '--samples', '1',
                        '--google-temperature', '0'], root=self.root)
        self.assertEqual(code, 0, error.getvalue())
        self.assertEqual(json.loads(output.getvalue())['status'], 'completed')
        folder = self.root / 'experiments/test/generations/gemini-cli'
        manifest = read_json(folder / 'manifest.json')
        self.assertEqual(manifest['settings']['provider_options']['api_key_env'], 'GEMINI_API_KEY')
        for path in folder.rglob('*'):
            if path.is_file():
                self.assertNotIn('unique-secret-for-test', path.read_text(encoding='utf-8'))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['generate', 'test', '--resume', 'gemini-cli', '--google-api-key-env', 'DIFFERENT_KEY'])
