"""Three-stage behavior with hand-annotated CoNLL-U and saved request fixtures."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from linguistic_eval._files import read_json, read_jsonl, write_json, write_text
from linguistic_eval.cli import main
from linguistic_eval.feature_extractor import extract_features, extract_folder
from linguistic_eval.parser_checks import check_constraints, check_folder, satisfaction_report
from linguistic_eval.parser_process import parse_folder

TEXT = 'Koulussa lapset lukevat kirjoja. Sähkövirta kulkee.'
CONLLU = '''# text = Koulussa lapset lukevat kirjoja.
1\tKoulussa\tkoulu\tNOUN\tN\tCase=Ine|Number=Sing\t3\tobl\t_\t_
2\tlapset\tlapsi\tNOUN\tN\tCase=Nom|Number=Plur\t3\tnsubj\t_\t_
3\tlukevat\tlukea\tVERB\tV\tTense=Pres\t0\troot\t_\t_
4\tkirjoja\tkirja\tNOUN\tN\tCase=Par|Number=Plur\t3\tobj\t_\t_
5\t.\t.\tPUNCT\tPunct\t_\t3\tpunct\t_\t_

# text = Sähkövirta kulkee.
1\tSähkövirta\tsähkö#virta\tNOUN\tN\tCase=Nom|Number=Sing\t2\tnsubj\t_\t_
2\tkulkee\tkulkea\tVERB\tV\tTense=Pres\t0\troot\t_\t_
3\t.\t.\tPUNCT\tPunct\t_\t2\tpunct\t_\t_

'''
CONSTRAINTS = {
    'vocabulary_constraints': {'items': ['koulu', 'kirja', 'sähkö#virta'], 'min_occurrences_per_item': 1},
    'linguistic_constraints': {'items': [{'control': 'Case=Par', 'occurrences': {'min': 1, 'max': 1}}]},
    'syntax_tree_constraints': {'items': [{'control': 'obj'}, {'control': 'root', 'occurrences': {'min': 2, 'max': 2}}]},
    'part_of_speech_constraints': {'items': [{'control': 'NOUN', 'proportion': {'min': 0.5, 'max': 0.5}}]},
    'length_constraints': {'number_of_words': {'min': 6, 'max': 6}, 'number_of_sentences': {'min': 2, 'max': 2}, 'max_word_length': 10},
}


class FakeParser:
    def __init__(self):
        self.calls = []
        self.conllu = CONLLU

    def parse(self, text):
        self.calls.append(text)
        if text == 'error':
            raise RuntimeError('Parser unavailable')
        return {'text': text, 'conllu': self.conllu, 'parser': {'backend': 'hand-annotated-fixture'}}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.experiment = self.root / 'experiments/X'
        write_json(self.experiment / 'template/template.json', {'constraints': {'wrong': 'mutable template'}})
        self.parser = FakeParser()
        self.generation = self.create_generation('Y')
        self.output = self.root / 'evaluations/deterministic_checks/X/Y'

    def create_generation(self, name, constraints=None, statuses=('completed',)):
        folder = self.experiment / 'generations' / name
        request = {'request_id': 'r1', 'configuration_id': 'config1', 'repetition_index': 1,
                   'conditions': {'constraints': deepcopy(constraints or CONSTRAINTS), 'word_length_unit': 'letters'}}
        write_json(folder / 'manifest.json', {'schema_version': '1.0', 'condition': {'constraints': {'wrong': {}}}})
        self.write_rows(folder / 'requests.jsonl', [request])
        self.write_rows(folder / 'results.jsonl', [
            {'request_id': 'r1', 'attempt_id': f'a{i}', 'status': status, 'text': TEXT,
             'configuration_id': 'config1', 'repetition_index': 1}
            for i, status in enumerate(statuses)
        ])
        return folder

    def write_rows(self, path, rows):
        write_text(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

    def run_stages(self, *, only_new=False):
        parsed = parse_folder(self.generation, parser=self.parser, only_new=only_new)
        extracted = extract_folder(self.output / 'conllus', only_new=only_new)
        checked = check_folder(self.output, only_new=only_new)
        return parsed, extracted, checked

    def reports(self):
        return read_jsonl(self.output / 'constraint_satisfaction.jsonl')

    def test_full_workflow_reports_passes_and_preserves_generation_inputs(self):
        before = {p: p.read_bytes() for p in self.generation.iterdir()}
        parsed, extracted, checked = self.run_stages()
        self.assertEqual((parsed[0]['parsed'], extracted['extracted'], checked['successes']), (1, 1, 1))
        record = self.reports()[0]
        self.assertEqual(record['success'], 1)
        self.assertEqual(set(record['satisfied_groups']), set(CONSTRAINTS))
        self.assertEqual(record['failed_features'], [])
        self.assertEqual(record['failure_reasons'], [])
        self.assertEqual(record['request_id'], 'r1')
        self.assertEqual(record['attempt_id'], 'a0')
        entry = read_json(self.output / 'constraints.json')['texts'][0]
        self.assertEqual(entry['constraints'], CONSTRAINTS)
        self.assertEqual(entry['word_length_unit'], 'characters')
        self.assertEqual({p: p.read_bytes() for p in self.generation.iterdir()}, before)

    def test_experiment_detection_single_generation_override_and_completed_attempts(self):
        self.generation = self.create_generation('Y', statuses=('completed', 'incomplete', 'failed', 'refused', 'completed'))
        self.create_generation('Z')
        outcomes = parse_folder(self.experiment, parser=self.parser)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0]['parsed'], 2)
        self.assertEqual(outcomes[0]['skipped_results'], 3)
        entries = read_json(self.output / 'constraints.json')['texts']
        self.assertEqual([e['attempt_id'] for e in entries], ['a0', 'a4'])
        self.assertEqual(len(list((self.output / 'conllus').glob('*.conllu'))), 2)
        custom = self.root / 'custom'
        parse_folder(self.generation, custom, parser=self.parser)
        self.assertTrue((custom / 'constraints.json').exists())
        custom_experiment = self.root / 'custom-experiment'
        parse_folder(self.experiment, custom_experiment, parser=self.parser)
        self.assertTrue((custom_experiment / 'Z/constraints.json').exists())

    def test_only_new_reuses_all_stages_and_default_replaces(self):
        self.run_stages()
        report_bytes = (self.output / 'constraint_satisfaction.jsonl').read_bytes()
        parsed, extracted, checked = self.run_stages(only_new=True)
        self.assertEqual((parsed[0]['reused'], extracted['reused'], checked['reused']), (1, 1, 1))
        self.assertEqual(len(self.parser.calls), 1)
        self.assertEqual((self.output / 'constraint_satisfaction.jsonl').read_bytes(), report_bytes)
        self.run_stages()
        self.assertEqual(len(self.parser.calls), 2)
        self.assertEqual(len(self.reports()), 1)
        rows = read_jsonl(self.generation / 'results.jsonl')
        rows.append({**rows[0], 'attempt_id': 'new-attempt'})
        self.write_rows(self.generation / 'results.jsonl', rows)
        parsed, extracted, checked = self.run_stages(only_new=True)
        self.assertEqual((parsed[0]['parsed'], extracted['extracted'], checked['checked']), (1, 1, 1))
        self.assertEqual(len(self.reports()), 2)

    def test_changed_conditions_recheck_without_reparsing_text(self):
        self.run_stages()
        requests = read_jsonl(self.generation / 'requests.jsonl')
        requests[0]['conditions']['constraints']['length_constraints']['max_word_length'] = 9
        self.write_rows(self.generation / 'requests.jsonl', requests)
        parsed, extracted, checked = self.run_stages(only_new=True)
        self.assertEqual((parsed[0]['reused'], extracted['reused'], checked['checked']), (1, 1, 1))
        report = self.reports()[0]
        self.assertEqual(report['success'], 0)
        self.assertEqual(report['failed_groups'], ['length_constraints'])
        self.assertEqual(report['failed_features'], ['max_word_length'])
        self.assertEqual(report['failure_reasons'][0]['realized_value'], 10)
        self.assertNotIn('length_constraints', report['satisfied_groups'])
        self.assertIn('number_of_words', report['satisfied_features'])

    def test_old_parser_processing_is_not_reused_by_only_new(self):
        self.run_stages()
        path = next((self.output / 'conllus').glob('*.json'))
        metadata = read_json(path)
        metadata.pop('parsing_version')
        write_json(path, metadata)
        self.assertEqual(check_folder(self.output, only_new=True)['errors'], 1)
        self.assertIn('rerun parse', self.reports()[0]['error'])
        parsed = parse_folder(self.generation, parser=self.parser, only_new=True)
        self.assertEqual(parsed[0]['parsed'], 1)
        self.assertEqual(len(self.parser.calls), 2)

    def test_changed_text_is_reparsed_and_stale_features_cannot_pass(self):
        self.run_stages()
        results = read_jsonl(self.generation / 'results.jsonl')
        results[0]['text'] += ' Uusi virke.'
        self.write_rows(self.generation / 'results.jsonl', results)
        self.parser.conllu = CONLLU + '1\tUusi\tuusi\tADJ\t_\t_\t0\troot\t_\t_\n\n'
        parsed = parse_folder(self.generation, parser=self.parser, only_new=True)
        self.assertEqual(parsed[0]['parsed'], 1)
        self.assertEqual(check_folder(self.output, only_new=True)['errors'], 1)
        self.assertEqual(self.reports()[0]['success'], 0)
        self.assertIn('stale', self.reports()[0]['error'])
        self.assertEqual(extract_folder(self.output / 'conllus', only_new=True)['extracted'], 1)
        self.assertEqual(check_folder(self.output, only_new=True)['errors'], 0)
        self.assertIn('number_of_sentences', self.reports()[0]['failed_features'])

    def test_parser_failure_and_missing_features_are_reported(self):
        parse_folder(self.generation, parser=self.parser)
        checked = check_folder(self.output)
        self.assertEqual(checked['errors'], 1)
        self.assertEqual(self.reports()[0]['success'], 0)
        results = read_jsonl(self.generation / 'results.jsonl')
        results[0]['text'] = 'error'
        self.write_rows(self.generation / 'results.jsonl', results)
        parsed = parse_folder(self.generation, parser=self.parser)
        self.assertEqual(parsed[0]['errors'], 1)
        self.assertEqual(list((self.output / 'conllus').glob('*.conllu')), [])
        check_folder(self.output)
        report = self.reports()[0]
        self.assertEqual(report['success'], 0)
        self.assertIn('Parser unavailable', report['error'])
        self.assertTrue(all(reason['realized_value'] is None for reason in report['failure_reasons']))
        cached = check_folder(self.output, only_new=True)
        self.assertEqual((cached['reused'], cached['errors']), (1, 1))

    def test_per_request_lemma_targets_remain_separate(self):
        requests = read_jsonl(self.generation / 'requests.jsonl')
        second = deepcopy(requests[0])
        second['request_id'] = 'r2'
        second['conditions']['constraints']['vocabulary_constraints']['items'] = ['puuttuva']
        self.write_rows(self.generation / 'requests.jsonl', requests + [second])
        results = read_jsonl(self.generation / 'results.jsonl')
        self.write_rows(self.generation / 'results.jsonl', results + [{**results[0], 'request_id': 'r2', 'attempt_id': 'a2'}])
        self.run_stages()
        reports = self.reports()
        self.assertEqual([r['success'] for r in reports], [1, 0])
        self.assertEqual(reports[1]['failed_features'], ['puuttuva'])

    def test_empty_completed_text_is_an_error_even_without_targets(self):
        requests = read_jsonl(self.generation / 'requests.jsonl')
        requests[0]['conditions']['constraints'] = {}
        self.write_rows(self.generation / 'requests.jsonl', requests)
        results = read_jsonl(self.generation / 'results.jsonl')
        results[0]['text'] = ''
        self.write_rows(self.generation / 'results.jsonl', results)
        parsed, extracted, checked = self.run_stages()
        self.assertEqual((parsed[0]['errors'], extracted['extracted'], checked['errors']), (1, 0, 1))
        self.assertEqual(self.parser.calls, [])
        self.assertEqual(self.reports()[0]['success'], 0)
        self.assertEqual(self.reports()[0]['failure_reasons'][0]['feature'], 'analysis')

    def test_removing_completed_result_removes_its_owned_parse_and_report(self):
        self.run_stages()
        rows = read_jsonl(self.generation / 'results.jsonl')
        rows[0]['status'] = 'failed'
        self.write_rows(self.generation / 'results.jsonl', rows)
        self.run_stages(only_new=True)
        self.assertEqual(self.reports(), [])
        self.assertEqual(list((self.output / 'conllus').glob('*.conllu')), [])

    def test_legacy_frozen_manifest_and_per_request_vocabulary(self):
        request = {'sample_id': 'old-sample', 'vocabulary': ['kirja']}
        result = {'sample_id': 'old-sample', 'attempt_id': 'old-attempt', 'status': 'completed', 'text': TEXT}
        write_json(self.generation / 'manifest.json', {'condition': {'constraints': CONSTRAINTS}})
        self.write_rows(self.generation / 'requests.jsonl', [request])
        self.write_rows(self.generation / 'results.jsonl', [result])
        self.run_stages()
        constraints = read_json(self.output / 'constraints.json')['texts'][0]['constraints']
        self.assertEqual(constraints['vocabulary_constraints']['items'], ['kirja'])
        self.assertEqual(self.reports()[0]['success'], 1)

    def test_cli_three_stages_and_only_new_avoid_loading_parser(self):
        def call(args):
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            return json.loads(output.getvalue())
        with patch('linguistic_eval.trankit_support.TrankitParser', return_value=self.parser):
            call(['parse', '--input-folder', str(self.generation)])
        call(['extract', '--input', str(self.output / 'conllus')])
        call(['check', '--input', str(self.output)])
        with patch('linguistic_eval.trankit_support.TrankitParser', return_value=FakeParser()) as constructor:
            self.assertEqual(call(['parse', '--input-folder', str(self.experiment), '--only-new'])[0]['reused'], 1)
            self.assertEqual(constructor.return_value.calls, [])
        self.assertEqual(self.reports()[0]['success'], 1)

    def test_source_folder_cannot_be_used_as_output(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            parse_folder(self.experiment, self.experiment, parser=self.parser)
        self.assertFalse((self.experiment / 'constraints.json').exists())


class FeatureAndCheckTests(unittest.TestCase):
    def test_spaced_word_forms_do_not_inflate_lengths_or_invent_linguistic_results(self):
        conllu = '1\tVanhan talon\tvanhantalo\tNOUN\t_\tCase=Gen\t0\troot\t_\t_\n'
        features = extract_features(conllu)
        self.assertEqual(features['length_constraints'],
                         {'number_of_words': 2, 'number_of_sentences': 1, 'max_word_length': 6})
        checks = check_constraints(features, {
            'length_constraints': {'max_word_length': 10},
            'vocabulary_constraints': {'items': ['talo']},
            'syntax_tree_constraints': {'items': ['root']},
        })
        self.assertEqual([c['status'] for c in checks], ['passed', 'unavailable', 'unavailable'])
        self.assertIsNone(checks[1]['realized_value'])

    def test_word_length_failure_reports_the_actual_forms(self):
        features = extract_features(CONLLU)
        report = satisfaction_report(check_constraints(features, {'length_constraints': {'max_word_length': 9}}))
        self.assertEqual(report['failure_reasons'][0]['words'], [{'word': 'Sähkövirta', 'characters': 10}])

    def test_exact_tags_character_lengths_and_lemma_normalization(self):
        features = extract_features(CONLLU)
        self.assertEqual(features['length_constraints'], {'number_of_words': 6, 'number_of_sentences': 2, 'max_word_length': 10})
        self.assertEqual(features['vocabulary_constraints']['sähkövirta'], 1)
        changed = deepcopy(CONSTRAINTS)
        changed['vocabulary_constraints']['items'] = ['Kirja', 'virta', 'sähkö#virta']
        changed['syntax_tree_constraints']['items'] = [{'control': 'acl'}, {'control': 'acl:relcl'}]
        features = extract_features(CONLLU.replace('\tobl\t', '\tacl:relcl\t'))
        report = satisfaction_report(check_constraints(features, changed))
        self.assertEqual(set(report['failed_features']), {'Kirja', 'virta', 'acl'})
        self.assertIn('sähkö#virta', report['satisfied_features'])
        self.assertIn('acl:relcl', report['satisfied_features'])
        multi = extract_features('1\ta-1\ta\tNOUN\t_\tCase=Acc,Nom\t0\troot\t_\t_\n')
        self.assertEqual(multi['length_constraints']['max_word_length'], 3)
        report = satisfaction_report(check_constraints(multi, {'linguistic_constraints': {'items': ['Case=Nom']}}))
        self.assertEqual(report['success'], 0)

    def test_mwt_headers_empty_nodes_and_extra_blank_lines_do_not_count(self):
        extra = '1-2\tMWT\t_\t_\t_\t_\t_\t_\t_\t_\n1.1\tx\tx\tX\t_\t_\t0\troot\t_\t_\n'
        actual = extract_features('\n\n' + extra + CONLLU + '\n\n')
        self.assertEqual(actual['token_count'], 8)
        self.assertEqual(actual['length_constraints']['number_of_sentences'], 2)

    def test_unsupported_features_missing_annotations_and_empty_groups(self):
        features = extract_features(CONLLU)
        checks = check_constraints(features, {
            'vocabulary_constraints': {'items': []}, 'length_constraints': {'number_of_words': None},
            'syntax_tree_constraints': {'items': [{'control': 'MaxTreeDepth', 'target': {'min': 0, 'max': 5}}]},
        })
        self.assertEqual(checks[0]['status'], 'unsupported')
        report = satisfaction_report(checks)
        self.assertEqual(report['success'], 0)
        self.assertEqual(report['satisfied_groups'], [])
        features = extract_features(CONLLU.replace('\tkirja\t', '\t_\t'))
        checks = check_constraints(features, CONSTRAINTS)
        vocabulary = [c for c in checks if c['group'] == 'vocabulary_constraints']
        self.assertTrue(all(c['status'] == 'unavailable' for c in vocabulary))

    def test_bad_conllu_creates_error_without_stopping_other_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'conllus'
            write_text(folder / 'valid.conllu', CONLLU)
            write_text(folder / 'broken.conllu', 'malformed')
            result = extract_folder(folder)
            self.assertEqual((result['extracted'], result['errors']), (1, 1))
            self.assertEqual(read_json(folder.parent / 'feature_analysis/broken.json')['status'], 'error')


if __name__ == '__main__':
    unittest.main()
