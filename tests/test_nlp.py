"""Offline interface checks plus an opt-in test of the real pinned parser."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from linguistic_eval.trankit_support import ParserError, TrankitParser
from linguistic_eval.trankit_support import trankit_resources as resources
from linguistic_eval.trankit_support._trankit_worker import split_whitespace_tokens


class ParserTests(unittest.TestCase):
    def test_whitespace_word_boundaries_are_general_and_preserve_source_spans(self):
        for form in ('Vanhan talon', 'kaksi\tasiaa', 'ääni\u00a0kuuluu', 'yksi\nkaksi\nkolme'):
            with self.subTest(form=form):
                sentence = {'id': 1, 'tokens': [{'id': 1, 'text': form, 'dspan': (10, 10 + len(form)),
                                                'span': (0, len(form))}, {'id': 2, 'text': '.'}]}
                repairs = split_whitespace_tokens([sentence])
                tokens = sentence['tokens']
                self.assertEqual([t['text'] for t in tokens], form.split() + ['.'])
                self.assertEqual([t['id'] for t in tokens], list(range(1, len(tokens) + 1)))
                self.assertEqual(len(repairs), 1)
                for token in tokens[:-1]:
                    start, end = token['span']
                    self.assertEqual(form[start:end], token['text'])
                    self.assertEqual(token['dspan'], (10 + start, 10 + end))

    def test_boundary_repair_preserves_mwt_expansions_and_renumbers_ranges(self):
        sentence = {'id': 1, 'tokens': [
            {'id': 1, 'text': 'kaksi sanaa'},
            {'id': (2, 3), 'text': 'jottei', 'expanded': [{'id': 2, 'text': 'jotta'}, {'id': 3, 'text': 'ei'}]},
            {'id': 4, 'text': '.'},
        ]}
        split_whitespace_tokens([sentence])
        tokens = sentence['tokens']
        self.assertEqual([t['id'] for t in tokens], [1, 2, (3, 4), 5])
        self.assertEqual(tokens[2]['text'], 'jottei')
        self.assertEqual([w['id'] for w in tokens[2]['expanded']], [3, 4])
        self.assertEqual([w['text'] for w in tokens[2]['expanded']], ['jotta', 'ei'])
        self.assertEqual(split_whitespace_tokens([sentence]), [])

    def test_import_does_not_load_nlp_libraries(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import linguistic_eval.trankit_support; '
                        'assert not {"torch", "trankit", "transformers", "numpy"} & sys.modules.keys()'],
                       check=True)

    def test_empty_input_and_missing_setup_fail_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.Popen') as spawn:
            parser = TrankitParser(cache_dir=tmp)
            with self.assertRaises(ValueError):
                parser.parse('  ')
            with self.assertRaisesRegex(ParserError, 'setup_trankit.py'):
                parser.parse('Hei.')
            spawn.assert_not_called()

    def test_worker_reuse_offline_environment_and_unicode_protocol(self):
        worker = Mock()
        worker.stdout = io.StringIO(''.join(json.dumps(row) + '\n' for row in (
            {'ready': {'device': 'cpu'}}, {'result': {'text': 'Hyvää päivää.'}},
            {'error': 'Analysis unavailable'}, {'result': {'text': 'Uusi teksti.'}},
        )))
        worker.stdin = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'resources.json').write_text('{}')
            with patch('subprocess.Popen', return_value=worker) as spawn:
                with TrankitParser(cache_dir=tmp, python='worker-python') as parser:
                    self.assertEqual(parser.parse('Hyvää päivää.')['text'], 'Hyvää päivää.')
                    with self.assertRaisesRegex(ParserError, 'Analysis unavailable'):
                        parser.parse('Virhe.')
                    self.assertEqual(parser.parse('Uusi teksti.')['text'], 'Uusi teksti.')
                    sent = worker.stdin.getvalue().splitlines()
                    self.assertEqual(json.loads(sent[0]), {'text': 'Hyvää päivää.'})
                spawn.assert_called_once()
                self.assertEqual(spawn.call_args.args[0][0], 'worker-python')
                self.assertEqual(spawn.call_args.args[0][2],
                                 'linguistic_eval.trankit_support._trankit_worker')
                env = spawn.call_args.kwargs['env']
                self.assertEqual(env['HF_HUB_OFFLINE'], '1')
                self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
                self.assertEqual(env['PYTHONHASHSEED'], '0')
                self.assertEqual(Path(env['PYTHONPATH']), Path(__file__).resolve().parents[1] / 'src')
                self.assertTrue(worker.stdin.closed)
                self.assertTrue(worker.stdout.closed)

    def test_default_worker_location_survives_nested_backend_package(self):
        root = Path(__file__).resolve().parents[1]
        relative = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
        with patch.dict(os.environ, {'CTRL_FIN_GEN_TRANKIT_PYTHON': ''}):
            self.assertEqual(Path(TrankitParser().python), root / '.venv-trankit' / relative)

    def test_startup_error_closes_worker(self):
        worker = Mock(stdout=io.StringIO('{"error": "Checksum mismatch"}\n'), stdin=io.StringIO())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'resources.json').write_text('{}')
            with patch('subprocess.Popen', return_value=worker):
                with self.assertRaisesRegex(ParserError, 'Checksum mismatch'):
                    TrankitParser(cache_dir=tmp).start()
        self.assertTrue(worker.stdin.closed)
        worker.wait.assert_called_once()

    def test_resource_integrity_and_missing_mwt_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            task_dir = cache / resources.EMBEDDING / 'finnish'
            snapshot = cache / resources.EMBEDDING / f'models--{resources.EMBEDDING}' / 'snapshots' / resources.ENCODER_REVISION
            task_dir.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            files = [task_dir / name for name in resources.TASK_FILES]
            files += [snapshot / name for name in resources.ENCODER_FILES]
            for path in files:
                path.write_text('fixture')
            ref = snapshot.parent.parent / 'refs/main'
            ref.parent.mkdir()
            ref.write_text(resources.ENCODER_REVISION)
            (task_dir / 'finnish.downloaded').touch()
            manifest = {
                'trankit_commit': resources.TRANKIT_COMMIT,
                'model_revision': resources.MODEL_REVISION,
                'encoder_revision': resources.ENCODER_REVISION,
                'sha256': {str(path.relative_to(cache)): resources.sha256(path) for path in files},
            }
            (cache / 'resources.json').write_text(json.dumps(manifest))
            self.assertEqual(resources.verify_resources(cache), manifest)
            (task_dir / 'finnish_mwt_expander.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                resources.verify_resources(cache)
            (task_dir / 'finnish_mwt_expander.pt').write_text('fixture')
            files[0].write_text('modified')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                resources.verify_resources(cache)


@unittest.skipUnless(os.environ.get('CTRL_FIN_GEN_TEST_TRANKIT') == '1',
                     'Set CTRL_FIN_GEN_TEST_TRANKIT=1 after running scripts/setup_trankit.py.')
class LiveParserTests(unittest.TestCase):
    def test_real_large_tdt_annotations_and_repeatability(self):
        text = 'Koulussa lapset lukevat kirjoja. Opettaja auttaa heitä.'
        with TrankitParser() as parser:
            first = parser.parse(text)
            self.assertEqual(first, parser.parse(text))
            # Interleave another document to catch accidental state leakage.
            parser.parse('Minä luen kirjaa, jonka opettaja antoi minulle.')
            self.assertEqual(first, parser.parse(text))
        with TrankitParser() as parser:
            self.assertEqual(first, parser.parse(text))
        self.assertEqual(first['text'], text)
        self.assertEqual(first['parser']['resources']['treebank'], 'UD_Finnish-TDT')
        self.assertEqual(first['parser']['resources']['embedding'], 'xlm-roberta-large')
        self.assertEqual(first['parser']['device'], 'cpu')
        self.assertEqual(first['parser']['tokenization_policy'], 'split_whitespace_word_forms_v1')
        sentences = first['analysis']['sentences']
        self.assertEqual(len(sentences), 2)
        tokens = sentences[0]['tokens']
        self.assertEqual([t['lemma'] for t in tokens[:4]], ['koulu', 'lapsi', 'lukea', 'kirja'])
        self.assertIn('Case=Ine', tokens[0]['feats'])
        self.assertEqual(tokens[3]['deprel'], 'obj')
        for sentence in sentences:
            words = [word for token in sentence['tokens']
                     for word in token.get('expanded', [token])]
            self.assertEqual(sum(word['head'] == 0 for word in words), 1)
        self.assertIn('\tkirja\tNOUN\t', first['conllu'])


if __name__ == '__main__':
    unittest.main()
