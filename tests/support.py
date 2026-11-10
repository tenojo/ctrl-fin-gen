"""Small fixtures shared by the behavioral tests. No API requests are made."""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from ctrl_fin_gen.prepare import create_experiment


REPOSITORY = Path(__file__).resolve().parents[1]


def response(response_id='resp_1', text='Koulussa luettiin kirjaa.', status='completed'):
    return {
        'id': response_id, 'model': 'test-model', 'status': status,
        'output': [{'type': 'message', 'role': 'assistant', 'content': [
            {'type': 'output_text', 'text': text},
        ]}],
        'usage': {'input_tokens': 20, 'output_tokens': 8},
    }


from ctrl_fin_gen.models import (GenerationSettings, GenerationResult, ProviderEvent,
                                 BatchJob, RecoveryReport)
from ctrl_fin_gen.providers.base import Provider, ProviderError


class FakeProvider(Provider):
    """Deliberately different service semantics: no SDK fields or native IDs."""
    streaming = True
    batch_jobs = True

    def __init__(self, settings=None, workspace='.', actions=None):
        super().__init__(settings or GenerationSettings('fake', 'test-model', 500), workspace)
        self.actions = list(actions or [])
        self.calls = []
        self.recoveries = {}
        self.saved = {}
        self.submissions = 0
        self.job = BatchJob('queued', {'alien_ticket': 41})
        self.batch_results = []

    def validate_settings(self):
        pass

    def format_request(self, request):
        return ('ALIEN_PROMPT', request.user_prompt)

    def format_result(self, native, attempt):
        r = attempt.request
        return GenerationResult(attempt.submission_id, r.request_id, attempt.attempt_id,
                                r.configuration_id, r.repetition_index,
                                native.get('status', 'completed'), native.get('text', 'Koulussa luettiin kirjaa.'))

    def generate(self, attempt, on_event):
        self.calls.append(attempt)
        on_event(ProviderEvent('started', attempt.attempt_id))
        action = self.actions.pop(0) if self.actions else {}
        if callable(action):
            return action(attempt, on_event)
        if self.settings.mode == 'stream':
            on_event(ProviderEvent('text_delta', attempt.attempt_id, text='Koulussa '))
        result = self.format_result(action, attempt)
        self.saved[attempt.attempt_id] = result
        return result

    def recover_request(self, attempt, checkpoint):
        if attempt.attempt_id in self.saved:
            return RecoveryReport('finished', result=self.saved[attempt.attempt_id])
        return self.recoveries.get(attempt.request.request_id, RecoveryReport('unknown'))

    def submit_batch(self, attempts, submission_id, on_event):
        self.submissions += 1
        self.batch_results = [self.format_result({}, a) for a in reversed(attempts)]
        return self.job

    def get_batch_status(self, job):
        return self.job

    def collect_batch(self, job, attempts, on_event):
        return iter(self.batch_results if self.job.status == 'completed' else [])

    def recover_batch(self, submission_id, checkpoint):
        return RecoveryReport('finished', job=self.job)


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(REPOSITORY / 'conditions', self.root / 'conditions')
        self.template = self.root / 'conditions/system_templates/template_pilot/template.json'

    def create(self, name='test', values=None, template='template_pilot', copy_from=None):
        return create_experiment(self.root, name, template, values, copy_from)

    def settings(self, **overrides):
        return {
            'backend': 'openai', 'model': 'test-model', 'samples': 2,
            'max_output_tokens': 500, 'stream': False, 'batch': False,
            **overrides,
        }
