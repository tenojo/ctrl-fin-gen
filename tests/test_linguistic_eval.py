"""Evaluation boundaries and CLI behavior, without models or generated artifacts."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from linguistic_eval.args import parse_args
from linguistic_eval.cli import main
from linguistic_eval.trankit_support import ParserError


class EvaluationInterfaceTests(unittest.TestCase):
    def test_evaluation_imports_without_generation_or_model_libraries(self):
        subprocess.run([sys.executable, '-c', '''
import importlib.abc
import sys
class BlockGeneration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'ctrl_fin_gen' or fullname.startswith('ctrl_fin_gen.'):
            raise AssertionError('Evaluation imported generation: ' + fullname)
sys.meta_path.insert(0, BlockGeneration())
import linguistic_eval.cli
import linguistic_eval.parser_process
import linguistic_eval.feature_extractor
import linguistic_eval.parser_checks
import linguistic_eval.trankit_support
assert not {'torch', 'trankit', 'transformers', 'numpy'} & sys.modules.keys()
'''], check=True)

    def test_generation_does_not_import_evaluation(self):
        subprocess.run([sys.executable, '-c', '''
import importlib.abc
import sys
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'linguistic_eval' or fullname.startswith('linguistic_eval.'):
            raise AssertionError('Generation imported evaluation: ' + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from ctrl_fin_gen.cli import parse_args
assert parse_args(['status', 'experiment', 'generation']).command == 'status'
'''], check=True)

    def test_missing_inputs_fail_before_model_start_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = str(root / 'missing')
            output = str(root / 'output')
            commands = [
                ['parse', '--input-folder', missing, '--output-folder', output],
                ['extract', '--input', missing, '--output-folder', output],
                ['check', '--input', missing, '--output', output],
            ]
            with patch('linguistic_eval.trankit_support.TrankitParser') as backend:
                for command in commands:
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with self.subTest(command=command), redirect_stdout(stdout), redirect_stderr(stderr):
                        self.assertEqual(main(command), 1)
                        self.assertEqual(stdout.getvalue(), '')
                        self.assertIn('Evaluation failed', stderr.getvalue())
                backend.return_value.__enter__.return_value.parse.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_parse_text_retains_unicode_json_and_conllu_outputs(self):
        result = {'text': 'Hyvää päivää.', 'conllu': '1\tHyvää\thyvä\tADJ\t_\t_\t0\troot\t_\t_\n\n'}
        backend = Mock()
        backend.__enter__ = Mock(return_value=backend)
        backend.__exit__ = Mock(return_value=False)
        backend.parse.return_value = result
        with patch('linguistic_eval.trankit_support.TrankitParser', return_value=backend):
            for format in ('json', 'conllu'):
                output = io.StringIO()
                with self.subTest(format=format), redirect_stdout(output):
                    self.assertEqual(main(['parse-text', '--text', result['text'], '--format', format]), 0)
                actual = output.getvalue()
                self.assertEqual(json.loads(actual) if format == 'json' else actual,
                                 result if format == 'json' else result['conllu'])
            backend.parse.assert_called_with(result['text'])

    def test_parser_error_is_not_reported_as_a_constraint_failure(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch('linguistic_eval.trankit_support.TrankitParser', side_effect=ParserError('Missing model')):
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(main(['parse-text', '--text', 'Hei.']), 1)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('Missing model', errors.getvalue())

    def test_cli_rejects_ambiguous_source_or_invalid_threads(self):
        for arguments in (
            ['parse', '--output-folder', 'out'],
            ['parse', '--input-folder', 'e', '--threads', '0'],
            ['parse-text', '--threads', '0'],
            ['parse-text', '--text', 'Hei.', '--input', 'file.txt'],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(arguments)
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
