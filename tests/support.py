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


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.retrieved = []
        self.remote_responses = {}
        self.job = {'id': 'batch_1', 'status': 'validating'}
        self.submissions = []
        self.files = {}

    def generate(self, body, stream, on_event):
        self.calls.append(copy.deepcopy(body))
        action = self.responses.pop(0) if self.responses else response(f'resp_{len(self.calls)}')
        if callable(action):
            return action(body, stream, on_event)
        result = copy.deepcopy(action)
        if stream:
            on_event({'type': 'response.created', 'response': {'id': result['id'], 'status': 'in_progress'}})
            on_event({'type': 'response.output_text.delta', 'delta': 'Koulussa '})
        on_event({'type': 'response.' + result['status'], 'response': result})
        return result

    def retrieve_response(self, response_id):
        self.retrieved.append(response_id)
        result = self.remote_responses[response_id]
        return copy.deepcopy(result.pop(0) if isinstance(result, list) else result)

    def submit_batch(self, path, submission_id):
        self.submissions.append(path.read_text(encoding='utf-8'))
        self.job['metadata'] = {'submission_id': submission_id}
        return copy.deepcopy(self.job)

    def find_batch(self, submission_id):
        return copy.deepcopy(self.job) if self.job.get('metadata', {}).get('submission_id') == submission_id else None

    def retrieve_batch(self, batch_id):
        assert batch_id == self.job['id']
        return copy.deepcopy(self.job)

    def download(self, file_id):
        return self.files[file_id]


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
