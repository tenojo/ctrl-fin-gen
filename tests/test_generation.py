import json
from openai import OpenAIError

from ctrl_fin_gen.generation import collect_results, generation_folder, get_status, resume_generation, start_generation
from ctrl_fin_gen.storage import append_jsonl, read_json, read_jsonl, submission_lock, write_json
from support import FakeProvider, WorkspaceTest, response


def interrupt_before_response(body, stream, on_event):
    raise ConnectionError('Connection lost before response ID was received')


def interrupt_after_response_id(body, stream, on_event):
    on_event({'type': 'response.created', 'response': {'id': 'resp_lost', 'status': 'in_progress'}})
    on_event({'type': 'response.output_text.delta', 'delta': 'Osittainen teksti'})
    raise ConnectionError('Connection lost')


class GenerationTests(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.create()

    def folder(self, name='run'):
        return generation_folder(self.root, 'test', name)

    def test_immediate_results_and_metadata_are_saved(self):
        provider = FakeProvider()
        state = start_generation(self.root, 'test', 'run', self.settings(reasoning_effort='low'), provider)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['finished_samples'], 2)
        records = read_jsonl(self.folder() / 'results.jsonl')
        self.assertEqual([record['text'] for record in records], ['Koulussa luettiin kirjaa.'] * 2)
        self.assertEqual(provider.calls[0]['reasoning'], {'effort': 'low'})
        self.assertEqual(provider.calls[0]['max_output_tokens'], 500)
        self.assertEqual(len({record['sample_id'] for record in records}), 2)
        self.assertEqual(len(list((self.folder() / 'raw').glob('*.json'))), 2)
        manifest = read_json(self.folder() / 'manifest.json')
        self.assertEqual(manifest['condition']['artifact_type'], 'realized_template')
        self.assertEqual(manifest['settings']['model'], 'test-model')
        with self.assertRaises(FileExistsError):
            start_generation(self.root, 'test', 'run', self.settings(), provider)
        resume_generation(self.root, 'test', 'run', provider)
        self.assertEqual(len(provider.calls), 2)

    def test_incomplete_refused_and_failed_outputs_are_kept(self):
        incomplete = response(text='Kesken jäänyt', status='incomplete')
        incomplete['incomplete_details'] = {'reason': 'max_output_tokens'}
        refused = response('resp_refusal')
        refused['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'En voi toteuttaa pyyntöä.'}]
        failed = response('resp_failed', text='', status='failed')
        failed['error'] = {'code': 'server_error', 'message': 'Temporary error'}
        provider = FakeProvider([incomplete, refused, failed])
        state = start_generation(self.root, 'test', 'run', self.settings(samples=3), provider)
        results = read_jsonl(self.folder() / 'results.jsonl')
        self.assertEqual([row['status'] for row in results], ['incomplete', 'refused', 'failed'])
        self.assertEqual(results[0]['text'], 'Kesken jäänyt')
        self.assertEqual(results[1]['refusal'], 'En voi toteuttaa pyyntöä.')
        self.assertEqual(state['status'], 'completed_with_errors')
        self.assertEqual(state['finished_samples'], 3)

    def test_stream_events_and_each_finished_result_survive_interruption(self):
        text = []
        provider = FakeProvider([response(), interrupt_after_response_id])
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            start_generation(self.root, 'test', 'run', self.settings(stream=True), provider, text.append)
        self.assertEqual(len(read_jsonl(self.folder() / 'results.jsonl')), 1)
        self.assertIn('Osittainen teksti', ''.join(text))
        events = read_jsonl(self.folder() / 'raw/events.jsonl')
        self.assertEqual(events[-1]['event']['type'], 'response.output_text.delta')
        provider.remote_responses['resp_lost'] = response('resp_lost', text='Palautettu teksti')
        state = resume_generation(self.root, 'test', 'run', provider)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.retrieved, ['resp_lost'])
        self.assertEqual(read_jsonl(self.folder() / 'results.jsonl')[1]['text'], 'Palautettu teksti')

    def test_waiting_response_is_retrieved_again_on_later_resume(self):
        provider = FakeProvider([interrupt_after_response_id])
        with self.assertRaises(RuntimeError):
            start_generation(self.root, 'test', 'run', self.settings(samples=1, stream=True), provider)
        provider.remote_responses['resp_lost'] = [response('resp_lost', status='in_progress'), response('resp_lost')]
        waiting = resume_generation(self.root, 'test', 'run', provider)
        self.assertEqual(waiting['status'], 'waiting_for_response')
        done = resume_generation(self.root, 'test', 'run', provider)
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.retrieved), 2)

    def test_unavailable_response_requires_explicit_retry(self):
        provider = FakeProvider([interrupt_after_response_id])
        with self.assertRaises(RuntimeError):
            start_generation(self.root, 'test', 'run', self.settings(samples=1, stream=True), provider)
        def unavailable(response_id):
            raise OpenAIError('Response is no longer retrievable')
        provider.retrieve_response = unavailable
        with self.assertRaises(OpenAIError):
            resume_generation(self.root, 'test', 'run', provider)
        state = resume_generation(self.root, 'test', 'run', provider, retry_unknown=True)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(len(provider.calls), 2)

    def test_unknown_outcome_needs_explicit_retry_and_keeps_saved_requests(self):
        provider = FakeProvider([interrupt_before_response])
        with self.assertRaises(RuntimeError):
            start_generation(self.root, 'test', 'run', self.settings(samples=1), provider)
        original = read_jsonl(self.folder() / 'requests.jsonl')
        with self.assertRaisesRegex(ValueError, 'unknown remote outcome'):
            resume_generation(self.root, 'test', 'run', provider)
        self.assertEqual(len(provider.calls), 1)
        # Resume uses frozen requests even if the experiment's source has changed.
        self.template.write_text('{}')
        state = resume_generation(self.root, 'test', 'run', provider, retry_unknown=True)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(read_jsonl(self.folder() / 'requests.jsonl'), original)
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertEqual(read_jsonl(self.folder() / 'attempts.jsonl')[2]['event'], 'abandoned')

    def test_terminal_event_survives_failure_before_raw_checkpoint(self):
        provider = FakeProvider([interrupt_before_response])
        with self.assertRaises(RuntimeError):
            start_generation(self.root, 'test', 'run', self.settings(samples=1), provider)
        active = read_json(self.folder() / 'state.json')['active_request']
        append_jsonl(self.folder() / 'raw/events.jsonl', {
            'sample_id': active['sample_id'], 'attempt_id': active['attempt_id'],
            'event': {'type': 'response.completed', 'response': response('resp_recovered')},
        })
        state = resume_generation(self.root, 'test', 'run', provider)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(len(provider.calls), 1)

    def test_collect_immediate_recovers_saved_raw_and_is_repeat_safe(self):
        provider = FakeProvider()
        start_generation(self.root, 'test', 'run', self.settings(), provider)
        (self.folder() / 'results.jsonl').unlink()
        state = collect_results(self.root, 'test', 'run')
        self.assertEqual(state['finished_samples'], 2)
        collect_results(self.root, 'test', 'run')
        self.assertEqual(len(read_jsonl(self.folder() / 'results.jsonl')), 2)

    def test_truncated_jsonl_tail_is_backed_up_before_resume(self):
        provider = FakeProvider([response(), interrupt_before_response])
        with self.assertRaises(RuntimeError):
            start_generation(self.root, 'test', 'run', self.settings(), provider)
        path = self.folder() / 'results.jsonl'
        with path.open('ab') as output:
            output.write(b'{"sample_id": "unfin')
        state = resume_generation(self.root, 'test', 'run', provider, retry_unknown=True)
        self.assertEqual(state['finished_samples'], 2)
        self.assertIn(b'unfin', path.with_name('results.jsonl.partial').read_bytes())

    def test_submission_lock_prevents_concurrent_resume(self):
        provider = FakeProvider()
        start_generation(self.root, 'test', 'run', self.settings(), provider)
        with submission_lock(self.folder()), self.assertRaisesRegex(ValueError, 'locked'):
            resume_generation(self.root, 'test', 'run', provider)
        self.assertFalse((self.folder() / '.lock').exists())

    def test_other_backends_fail_before_creating_a_submission(self):
        with self.assertRaisesRegex(ValueError, 'Only the OpenAI'):
            start_generation(self.root, 'test', 'run', self.settings(backend='google'), FakeProvider())
        self.assertFalse(self.folder().exists())


class BatchTests(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.create()
        self.provider = FakeProvider()
        start_generation(self.root, 'test', 'batch', self.settings(batch=True), self.provider)
        self.folder = generation_folder(self.root, 'test', 'batch')
        self.requests = read_jsonl(self.folder / 'requests.jsonl')

    def complete_job(self):
        self.provider.job.update({'status': 'completed', 'output_file_id': 'output', 'error_file_id': 'errors'})
        self.provider.files['output'] = json.dumps({
            'custom_id': self.requests[1]['sample_id'], 'response': {'status_code': 200, 'body': response()}, 'error': None,
        }) + '\n'
        self.provider.files['errors'] = json.dumps({
            'custom_id': self.requests[0]['sample_id'], 'response': None,
            'error': {'code': 'batch_expired', 'message': 'Request expired'},
        }) + '\n'

    def test_submitted_payload_and_status(self):
        entries = [json.loads(line) for line in self.provider.submissions[0].splitlines()]
        self.assertEqual([entry['custom_id'] for entry in entries], [row['sample_id'] for row in self.requests])
        self.assertTrue(all(entry['url'] == '/v1/responses' for entry in entries))
        self.assertTrue(all(entry['method'] == 'POST' for entry in entries))
        self.assertNotIn('stream', entries[0]['body'])
        status = get_status(self.root, 'test', 'batch', self.provider)
        self.assertEqual(status['provider_status'], 'validating')
        self.assertEqual(status['finished_samples'], 0)
        resume_generation(self.root, 'test', 'batch', self.provider)
        self.assertEqual(len(self.provider.submissions), 1)

    def test_out_of_order_success_and_error_collection_is_repeat_safe(self):
        self.complete_job()
        state = collect_results(self.root, 'test', 'batch', self.provider)
        results = read_jsonl(self.folder / 'results.jsonl')
        self.assertEqual(state['status'], 'completed_with_errors')
        self.assertEqual(results[0]['sample_id'], self.requests[1]['sample_id'])
        self.assertEqual(results[1]['error']['code'], 'batch_expired')
        collect_results(self.root, 'test', 'batch', self.provider)
        self.assertEqual(len(read_jsonl(self.folder / 'results.jsonl')), 2)
        self.assertEqual((self.folder / 'raw/error_file_id.jsonl').read_text(), self.provider.files['errors'])

    def test_unknown_custom_id_keeps_raw_but_does_not_attach_result(self):
        self.complete_job()
        self.provider.files['output'] = json.dumps({'custom_id': 'not-ours', 'response': {'body': response()}})
        with self.assertRaisesRegex(ValueError, 'unknown sample ID'):
            collect_results(self.root, 'test', 'batch', self.provider)
        self.assertTrue((self.folder / 'raw/output_file_id.jsonl').is_file())
        self.assertEqual(read_jsonl(self.folder / 'results.jsonl'), [])

    def test_missing_batch_results_are_reported_without_fabricating_outputs(self):
        self.complete_job()
        self.provider.job['error_file_id'] = None
        state = collect_results(self.root, 'test', 'batch', self.provider)
        self.assertEqual(state['provider_status'], 'completed')
        self.assertEqual(state['status'], 'missing_results')
        self.assertEqual(state['remaining_samples'], 1)
        self.assertEqual(len(read_jsonl(self.folder / 'results.jsonl')), 1)

    def test_submission_is_reconciled_using_metadata_after_lost_checkpoint(self):
        state = read_json(self.folder / 'state.json')
        state.update({'provider_job_id': None, 'status': 'submitting'})
        write_json(self.folder / 'state.json', state)
        state = resume_generation(self.root, 'test', 'batch', self.provider)
        self.assertEqual(state['provider_job_id'], 'batch_1')
        self.assertEqual(len(self.provider.submissions), 1)

    def test_unknown_batch_submission_requires_explicit_retry(self):
        state = read_json(self.folder / 'state.json')
        state.update({'provider_job_id': None, 'status': 'submitting'})
        write_json(self.folder / 'state.json', state)
        self.provider.job['metadata'] = {}
        with self.assertRaisesRegex(ValueError, 'unknown outcome'):
            resume_generation(self.root, 'test', 'batch', self.provider)
        resume_generation(self.root, 'test', 'batch', self.provider, retry_unknown=True)
        self.assertEqual(len(self.provider.submissions), 2)
