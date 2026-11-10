"""Contract-level tests: general orchestration never needs native provider shapes."""
from dataclasses import asdict, replace

from ctrl_fin_gen.generation import (start_generation, resume_generation, generation_folder,
                                    collect_results, get_status, save_result, progress)
from ctrl_fin_gen.models import GenerationSettings, RecoveryReport, ProviderEvent
from ctrl_fin_gen.providers.base import ProviderError
from ctrl_fin_gen.prepare import prepare_requests
from ctrl_fin_gen.storage import read_json, read_jsonl, write_json, submission_lock, append_jsonl
from support import WorkspaceTest, FakeProvider


def disconnect(attempt, events):
    events(ProviderEvent('checkpoint', attempt.attempt_id, checkpoint={'alien_cursor': 91}))
    raise ProviderError('Connection interrupted')


class GenerationTests(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.inputs = self.create()
        self.folder = generation_folder(self.root, 'test', 'run')

    def start(self, actions=None, samples=2, mode='immediate'):
        settings = GenerationSettings('fake', 'test-model', 500, mode)
        self.provider = FakeProvider(settings, self.folder / 'provider', actions)
        return start_generation(self.root, 'test', 'run', settings, samples, self.provider)

    def test_neutral_plan_and_metadata(self):
        state = self.start()
        self.assertEqual(state['status'], 'completed')
        requests = read_jsonl(self.folder / 'requests.jsonl')
        self.assertTrue(all('body' not in r and 'model' not in r for r in requests))
        self.assertEqual(requests[0]['configuration_id'], requests[1]['configuration_id'])
        self.assertEqual([r['repetition_index'] for r in requests], [1, 2])
        self.assertTrue(all('system_prompt' in r and 'user_prompt' in r for r in requests))
        self.assertEqual(read_json(self.folder / 'manifest.json')['schema_version'], '1.0')

    def test_known_terminal_outcomes_never_retried(self):
        self.start([{'status': 'incomplete', 'text': 'Kesken'}, {'status': 'refused'}])
        state = resume_generation(self.root, 'test', 'run', self.provider)
        self.assertEqual(state['status'], 'completed_with_errors')
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(read_jsonl(self.folder / 'results.jsonl')[0]['text'], 'Kesken')

    def test_unknown_needs_permission_and_frozen_requests_survive_source_change(self):
        with self.assertRaises(ProviderError):
            self.start([{}, disconnect])
        original = (self.folder / 'requests.jsonl').read_bytes()
        self.template.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unknown remote outcome'):
            resume_generation(self.root, 'test', 'run', self.provider)
        resume_generation(self.root, 'test', 'run', self.provider, retry_unknown=True)
        self.assertEqual((self.folder / 'requests.jsonl').read_bytes(), original)
        self.assertEqual(len(self.provider.calls), 3)
        self.assertEqual(len(read_jsonl(self.folder / 'results.jsonl')), 2)

    def test_waiting_then_recovered_without_regeneration(self):
        with self.assertRaises(ProviderError):
            self.start([disconnect], samples=1)
        attempt = self.provider.calls[0]
        self.provider.recoveries[attempt.request.request_id] = RecoveryReport('running')
        state = resume_generation(self.root, 'test', 'run', self.provider)
        self.assertEqual(state['status'], 'waiting_for_response')
        result = self.provider.format_result({'text': 'Palautettu'}, attempt)
        self.provider.recoveries[attempt.request.request_id] = RecoveryReport('finished', result=result)
        state = resume_generation(self.root, 'test', 'run', self.provider)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(len(self.provider.calls), 1)

    def test_safe_to_retry_does_not_require_unknown_override(self):
        with self.assertRaises(ProviderError):
            self.start([disconnect], samples=1)
        self.provider.recoveries['sample_000001'] = RecoveryReport('safe_to_retry')
        self.assertEqual(resume_generation(self.root, 'test', 'run', self.provider)['status'], 'completed')

    def test_every_attempt_retained_and_conflicts_rejected(self):
        self.start(samples=1)
        original = self.provider.saved[self.provider.calls[0].attempt_id]
        other = replace(original, attempt_id='another-attempt', status='failed')
        attempt = replace(self.provider.calls[0], attempt_id='another-attempt')
        append_jsonl(self.folder / 'attempts.jsonl', {**asdict(attempt), 'event': 'started'})
        save_result(self.folder, other)
        save_result(self.folder, other)
        state = progress(self.folder, read_json(self.folder / 'state.json'))
        self.assertEqual(state['finished_attempts'], 2)
        self.assertEqual(state['finished_samples'], 1)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            save_result(self.folder, replace(other, text='Changed'))
        with self.assertRaises(ValueError):
            save_result(self.folder, replace(other, request_id='unknown'))

    def test_collect_recovers_local_provider_evidence_repeatably(self):
        self.start()
        (self.folder / 'results.jsonl').unlink()
        collect_results(self.root, 'test', 'run', self.provider)
        collect_results(self.root, 'test', 'run', self.provider)
        self.assertEqual(len(read_jsonl(self.folder / 'results.jsonl')), 2)

    def test_tail_repair_and_lock(self):
        with self.assertRaises(ProviderError):
            self.start([{}, disconnect])
        path = self.folder / 'results.jsonl'
        with path.open('ab') as output:
            output.write(b'{"partial":')
        with submission_lock(self.folder), self.assertRaisesRegex(ValueError, 'locked'):
            resume_generation(self.root, 'test', 'run', self.provider)
        resume_generation(self.root, 'test', 'run', self.provider, retry_unknown=True)
        self.assertTrue(path.with_suffix('.jsonl.partial').exists())

    def test_reuse_identical_plan(self):
        self.start()
        plan = prepare_requests(self.inputs, 2)
        provider = FakeProvider()
        start_generation(self.root, 'test', 'copy', provider.settings, 2, provider, requests=plan)
        copied = generation_folder(self.root, 'test', 'copy')
        self.assertEqual(read_jsonl(self.folder / 'requests.jsonl'), read_jsonl(copied / 'requests.jsonl'))

    def test_legacy_rejected_without_mutation(self):
        self.folder.mkdir(parents=True)
        write_json(self.folder / 'manifest.json', {'settings': {'backend': 'openai'}})
        before = (self.folder / 'manifest.json').read_bytes()
        for operation in (resume_generation, get_status, collect_results):
            with self.assertRaisesRegex(ValueError, 'fresh run'):
                operation(self.root, 'test', 'run')
        self.assertEqual((self.folder / 'manifest.json').read_bytes(), before)
        self.assertFalse((self.folder / '.lock').exists())

    def test_batch_lifecycle_with_opaque_non_sdk_handle(self):
        self.start(mode='batch')
        self.assertEqual(get_status(self.root, 'test', 'run', self.provider)['provider_status'], 'queued')
        resume_generation(self.root, 'test', 'run', self.provider)
        self.assertEqual(self.provider.submissions, 1)
        self.provider.job.status = 'completed'
        state = collect_results(self.root, 'test', 'run', self.provider)
        self.assertEqual(state['status'], 'completed')
        collect_results(self.root, 'test', 'run', self.provider)
        self.assertEqual(len(read_jsonl(self.folder / 'results.jsonl')), 2)

    def test_batch_missing_and_unknown_submission(self):
        self.start(mode='batch')
        self.provider.job.status = 'completed'
        self.provider.batch_results.pop()
        self.assertEqual(collect_results(self.root, 'test', 'run', self.provider)['status'], 'missing_results')
        state = read_json(self.folder / 'state.json')
        state.update(job=None, status='submitting')
        write_json(self.folder / 'state.json', state)
        self.provider.recover_batch = lambda *args: RecoveryReport('unknown')
        with self.assertRaisesRegex(ValueError, 'unknown outcome'):
            resume_generation(self.root, 'test', 'run', self.provider)

    def test_unsupported_mode_before_files(self):
        settings = GenerationSettings('fake', 'm', 100, 'batch')
        provider = FakeProvider(settings)
        provider.batch_jobs = False
        with self.assertRaises(ValueError):
            start_generation(self.root, 'test', 'run', settings, 1, provider)
        self.assertFalse(self.folder.exists())

    def test_frozen_request_mutation_is_rejected(self):
        self.start()
        with (self.folder / 'requests.jsonl').open('a') as output:
            output.write('\n')
        with self.assertRaisesRegex(ValueError, 'Frozen request plan'):
            resume_generation(self.root, 'test', 'run', self.provider)

    def test_shared_checkpoint_replay_does_not_inspect_provider_keys(self):
        with self.assertRaises(ProviderError):
            self.start([disconnect], samples=1)
        state = read_json(self.folder / 'state.json')
        for entry in state['active'].values():
            entry['checkpoint'] = {}
        write_json(self.folder / 'state.json', state)
        def recover(attempt, checkpoint):
            self.assertEqual(checkpoint, {'alien_cursor': 91})
            return RecoveryReport('finished', result=self.provider.format_result({}, attempt))
        self.provider.recover_request = recover
        self.assertEqual(resume_generation(self.root, 'test', 'run', self.provider)['status'], 'completed')

    def test_uncertain_batch_retry_assigns_fresh_attempts(self):
        self.start(mode='batch')
        state = read_json(self.folder / 'state.json')
        before = {a['attempt_id'] for a in state['batch_attempts']}
        state.update(job=None, status='submitting')
        write_json(self.folder / 'state.json', state)
        self.provider.recover_batch = lambda *args: RecoveryReport('unknown')
        resume_generation(self.root, 'test', 'run', self.provider, retry_unknown=True)
        after = {a['attempt_id'] for a in read_json(self.folder / 'state.json')['batch_attempts']}
        self.assertFalse(before & after)
        self.assertEqual(self.provider.submissions, 2)

    def test_local_grouping_override_needs_no_runner_change(self):
        settings = GenerationSettings('fake', 'test-model', 500)
        provider = FakeProvider(settings)
        def grouped(attempts, events):
            for attempt in reversed(attempts):
                yield provider.generate(attempt, events)
        provider.generate_many = grouped
        state = start_generation(self.root, 'test', 'run', settings, 3, provider)
        self.assertEqual(state['finished_samples'], 3)
        self.assertEqual(read_jsonl(self.folder / 'results.jsonl')[0]['request_id'], 'sample_000003')
