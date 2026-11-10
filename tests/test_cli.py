import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from ctrl_fin_gen.cli import main, parse_args
from ctrl_fin_gen.storage import read_jsonl, write_json
from support import FakeProvider, WorkspaceTest


class CLITests(WorkspaceTest):
    def run_cli(self, arguments):
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(arguments, root=self.root)
        return code, output.getvalue(), errors.getvalue()

    def test_create_preview_then_generate_status_and_collect(self):
        create_args = ['create', 'science', '--template', 'third_grade_sciencebook_pilot',
                       '--lemmas', 'koulu', 'kirja', '--sentences', '5', '5']
        code, output, _ = self.run_cli(create_args + ['--preview'])
        self.assertEqual(code, 0)
        self.assertIn('koulu, kirja', json.loads(output)['messages'][-1]['content'])
        self.assertFalse((self.root / 'experiments/science').exists())
        self.assertEqual(self.run_cli(create_args)[0], 0)
        provider = FakeProvider()
        provider.settings.mode = 'stream'
        with patch('ctrl_fin_gen.generation.create_provider', return_value=provider):
            code, output, errors = self.run_cli([
                'generate', 'science', '--name', 'first', '--backend', 'openai',
                '--model', 'test-model', '--max-length', '500', '--samples', '2', '--stream',
            ])
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output)['status'], 'completed')
        self.assertIn('Koulussa', errors)
        self.assertEqual(self.run_cli(['status', 'science', 'first'])[0], 0)
        self.assertEqual(self.run_cli(['collect', 'science', 'first'])[0], 0)
        self.assertEqual(self.run_cli(['generate', 'science', '--resume', 'first'])[0], 0)
        self.assertEqual(len(provider.calls), 2)

    def test_prompt_file_and_empty_list_override(self):
        prompt = self.root / 'prompt.txt'
        prompt.write_text('Kirjoita veden kierrosta.', encoding='utf-8')
        code, output, errors = self.run_cli([
            'create', 'science', '--template', 'template_pilot', '--lemmas',
            '--prompt-file', str(prompt), '--audience', 'Lukiolaiset', '--preview',
        ])
        self.assertEqual(code, 0, errors)
        text = json.loads(output)['messages'][-1]['content']
        self.assertIn('veden kierrosta', text)
        self.assertIn('Lukiolaiset', text)
        self.assertNotIn('seuraavia sanoja', text)

    def test_generate_prompts_for_a_new_name_and_rejects_reuse(self):
        self.create()
        args = ['generate', 'test', '--backend', 'openai', '--model', 'test-model',
                '--max-length', '500', '--samples', '1']
        with patch('builtins.input', return_value='named'), patch('ctrl_fin_gen.generation.create_provider', return_value=FakeProvider()):
            self.assertEqual(self.run_cli(args)[0], 0)
            code, _, errors = self.run_cli(args)
        self.assertEqual(code, 1)
        self.assertIn('already exists', errors)

    def test_errors_are_clear_for_unsupported_backends_and_missing_credentials(self):
        self.create()
        args = ['generate', 'test', '--name', 'first', '--backend', 'vllm_python',
                '--model', 'test-model', '--max-length', '500', '--samples', '1']
        code, _, errors = self.run_cli(args)
        self.assertEqual(code, 1)
        self.assertIn('not implemented', errors)
        args[5] = 'openai'
        with patch.dict('os.environ', {}, clear=True):
            code, _, errors = self.run_cli(args)
        self.assertEqual(code, 1)
        self.assertIn('api_key', errors)
        self.assertNotIn('Traceback', errors)
        self.assertTrue((self.root / 'experiments/test/generations/first/requests.jsonl').exists())

    def test_resume_cannot_override_settings_and_retry_is_resume_only(self):
        cases = [
            ['generate', 'test', '--resume', 'run', '--model', 'other'],
            ['generate', 'test', '--resume', 'run', '--stream'],
            ['generate', 'test', '--retry-unknown'],
            ['generate', 'test', '--backend', 'openai', '--model', 'test-model', '--max-length', '-1', '--samples', '1'],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_foreign_provider_options_are_rejected_before_creating_a_submission(self):
        self.create()
        cases = [('google_cloud', ['--reasoning-effort', 'low']),
                 ('openai', ['--google-temperature', '0.5'])]
        for backend, flags in cases:
            with self.subTest(backend=backend):
                code, _, errors = self.run_cli([
                    'generate', 'test', '--name', 'wrong-flags', '--backend', backend,
                    '--model', 'model', '--samples', '1', '--max-length', '100', *flags,
                ])
                self.assertEqual(code, 1)
                self.assertIn('options cannot be used', errors)
                self.assertFalse((self.root / 'experiments/test/generations/wrong-flags').exists())

    def test_static_planning_submissions_get_a_clear_error(self):
        folder = self.root / 'experiments/old/generations/example'
        folder.mkdir(parents=True)
        write_json(folder / 'manifest.json', {'provider': {'backend': 'openai'}})
        code, _, errors = self.run_cli(['status', 'old', 'example'])
        self.assertEqual(code, 1)
        self.assertIn('static planning examples', errors)

    def test_sampling_cli_records_frozen_vocabulary(self):
        pool = self.root / 'pool.txt'
        pool.write_text('koulu\nkirja\ntaulu\npolku\n', encoding='utf-8')
        code, _, errors = self.run_cli([
            'create', 'science', '--template', 'template_pilot', '--vocabulary-pool', str(pool),
            '--sample-lemmas', '2', '--seed', '0', '--selection-scope', 'request',
        ])
        self.assertEqual(code, 0, errors)
        pool.unlink()
        with patch('ctrl_fin_gen.generation.create_provider', return_value=FakeProvider()):
            code, _, errors = self.run_cli([
                'generate', 'science', '--name', 'sampled', '--backend', 'openai',
                '--model', 'test-model', '--max-length', '500', '--samples', '3',
            ])
        self.assertEqual(code, 0, errors)
        requests = read_jsonl(self.root / 'experiments/science/generations/sampled/requests.jsonl')
        self.assertTrue(all(len(request['conditions']['constraints']['vocabulary_constraints']['items']) == 2 for request in requests))
        self.assertTrue(all(', '.join(request['conditions']['constraints']['vocabulary_constraints']['items']) in request['user_prompt'] for request in requests))

    def test_another_registered_provider_owns_its_flags_without_cli_changes(self):
        from ctrl_fin_gen.providers import PROVIDERS
        class AlienProvider(FakeProvider):
            @classmethod
            def add_cli_arguments(cls, parser):
                parser.add_argument('--alien-knob', type=int)
            @classmethod
            def options_from_args(cls, args):
                return {} if args.alien_knob is None else {'knob': args.alien_knob}
            def validate_settings(self):
                if self.settings.provider_options != {'knob': 7}:
                    raise ValueError('Missing native knob')
        self.create()
        with patch.dict(PROVIDERS, {'alien': AlienProvider}):
            code, output, errors = self.run_cli([
                'generate', 'test', '--name', 'alien', '--backend', 'alien',
                '--model', 'm', '--samples', '1', '--max-length', '100', '--alien-knob', '7',
            ])
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output)['status'], 'completed')
