import json

from ctrl_fin_gen.constraints.vocabulary import read_lemmas, vocabulary_for_request
from ctrl_fin_gen.prepare import create_experiment, load_experiment, load_values, prepare_experiment
from ctrl_fin_gen.prompts import build_messages
from ctrl_fin_gen.storage import read_json, write_json
from support import WorkspaceTest


def inline(items):
    return {'selection': {'mode': 'inline', 'items': items}}


class PreparationTests(WorkspaceTest):
    def test_prompt_order_and_finnish_instructions(self):
        prepared = self.create(values={
            'constraints': {
                'length_constraints': {'number_of_sentences': {'min': 5, 'max': 5}, 'max_word_length': 10},
                'vocabulary_constraints': inline(['koulu', 'kirja']),
                'linguistic_constraints': inline(['Case=Gen', 'Case=Par']),
                'syntax_tree_constraints': inline(['acl:relcl']),
            },
            'audience': {'description': 'Ammatillisen koulutuksen sähköalan opiskelijat'},
            'prompt': 'Kirjoita sähköstä.',
            'prompts': {'system': 'Vastaa suomeksi.', 'user_prefix': 'ALOITUS', 'user_reminder': 'LOPETUS'},
        })
        messages = build_messages(prepared)
        self.assertEqual(messages[0], {'role': 'system', 'content': 'Vastaa suomeksi.'})
        user = messages[1]['content']
        parts = ['ALOITUS', 'virkemäärän tulee olla 5', 'koulu, kirja', 'käyttää genetiiviä (Case=Gen)',
                 'käyttää relatiivilauseita', 'sähköalan opiskelijat', 'Kirjoita sähköstä.', 'LOPETUS']
        positions = [user.index(part) for part in parts]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('{', user)

    def test_experiment_reloads_without_copying_system_template(self):
        prepared = self.create()
        loaded = load_experiment(self.root, 'test')
        self.assertEqual(build_messages(loaded), build_messages(prepared))
        folder = self.root / 'experiments/test/template'
        self.assertEqual(read_json(folder / 'template.json')['artifact_type'], 'realized_template')
        self.assertTrue((folder / 'prompt_layout.txt').is_file())
        self.assertFalse((folder / 'system_template.json').exists())
        self.assertEqual(read_json(folder / 'control_definitions.json')['linguistic_constraints'], {})

    def test_user_science_template_and_empty_sections(self):
        prepared = self.create(template='third_grade_sciencebook_pilot')
        text = build_messages(prepared)[-1]['content']
        self.assertIn('3. luokan oppilaille', text)
        self.assertNotIn('Case=', text)
        self.assertNotIn('{', text)
        self.assertNotIn('part_of_speech_constraints', prepared.template['constraints'])

    def test_preview_is_read_only_and_names_are_unique(self):
        create_experiment(self.root, 'preview', 'template_pilot', preview=True)
        self.assertFalse((self.root / 'experiments').exists())
        self.create()
        with self.assertRaises(FileExistsError):
            self.create()
        with self.assertRaises(ValueError):
            self.create('../outside')

    def test_unsupported_controls_and_bounds_are_rejected_before_writing(self):
        cases = [
            {'constraints': {'invented': inline([])}},
            {'constraints': {'linguistic_constraints': inline(['NOT_A_FEATURE'])}},
            {'constraints': {'linguistic_constraints': inline(['Case=Gen', 'Case=Gen'])}},
            {'constraints': {'length_constraints': {'number_of_words': {'min': 20, 'max': 10}}}},
            {'constraints': {'length_constraints': {'max_word_length': 1000}}},
            {'audience': 3},
            {'word_length_unit': 'letters'},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.create(values=values)
        self.assertFalse((self.root / 'experiments/test').exists())

    def test_combined_morphosyntactic_count_limit(self):
        with self.assertRaisesRegex(ValueError, 'combined'):
            self.create(values={'constraints': {
                'linguistic_constraints': inline(['Case=Gen', 'Case=Par', 'Tense=Past']),
                'syntax_tree_constraints': inline(['acl:relcl', 'obj', 'nsubj']),
            }})

    def test_numeric_and_occurrence_targets(self):
        prepared = self.create(values={'constraints': {
            'syntax_tree_constraints': inline([{'control': 'MaxTreeDepth', 'target': {'min': 2, 'max': 4}}]),
            'linguistic_constraints': inline([{'control': 'Case=Gen', 'occurrences': {'min': 2, 'max': 3}}]),
        }})
        text = build_messages(prepared)[-1]['content']
        self.assertIn('2–3 kertaa', text)
        self.assertIn('2–4', text)
        with self.assertRaises(ValueError):
            prepare_experiment(self.root, 'template_pilot', {'constraints': {'syntax_tree_constraints': inline(['MaxTreeDepth'])}})

    def test_complexity_remains_an_explicit_future_feature(self):
        with self.assertRaisesRegex(ValueError, 'Parsebank'):
            self.create(values={'constraints': {'vocabulary_complexity_constraints': inline([
                {'control': 'mean_zipf', 'target': {'min': 2, 'max': 4}},
            ])}})

    def test_copy_preserves_values_and_applies_one_override(self):
        self.create('original', values={
            'constraints': {
                'length_constraints': {'number_of_words': {'min': 40, 'max': 50}},
                'vocabulary_constraints': inline(['koulu']),
            },
            'prompts': {'user_prefix': 'ALKU', 'user_reminder': 'LOPPU'},
        })
        copied = self.create('copy', copy_from='original', values={
            'constraints': {'length_constraints': {'number_of_sentences': {'min': 4, 'max': 4}}},
            'prompts': {'user_prefix': 'UUSI ALKU'},
        })
        self.assertEqual(copied.template['constraints']['length_constraints']['number_of_words']['min'], 40)
        self.assertEqual(copied.template['constraints']['vocabulary_constraints']['items'], ['koulu'])
        self.assertEqual(copied.prompts['user_reminder'], 'LOPPU')
        self.assertEqual(copied.prompts['user_prefix'], 'UUSI ALKU')
        with self.assertRaisesRegex(ValueError, 'same system-template'):
            self.create('wrong', template='third_grade_sciencebook_pilot', copy_from='original')

    def test_modified_catalogue_and_template_require_new_experiment(self):
        self.create()
        catalogue = self.root / 'conditions/linguistic_controls/length_controls.json'
        catalogue.write_text(catalogue.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'catalogue has changed'):
            load_experiment(self.root, 'test')
        self.template.write_text(self.template.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'system template has changed'):
            load_experiment(self.root, 'test')

    def test_bad_layout_fails_before_creating_folder(self):
        layout = self.template.parent / 'prompt_layout.txt'
        layout.write_text('{unknown_placeholder}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            self.create()
        self.assertFalse((self.root / 'experiments/test').exists())

    def test_condition_relative_paths(self):
        folder = self.root / 'inputs'
        folder.mkdir()
        (folder / 'lemmas.txt').write_text('koulu\nkirja\n', encoding='utf-8')
        write_json(folder / 'condition.json', {'constraints': {
            'vocabulary_constraints': {'selection': {'mode': 'file', 'path': 'lemmas.txt'}},
        }})
        prepared = self.create(values=load_values(folder / 'condition.json'))
        self.assertEqual(prepared.template['constraints']['vocabulary_constraints']['items'], ['koulu', 'kirja'])

    def test_sampling_is_frozen_and_copy_needs_no_source_file(self):
        pool = self.root / 'pool.txt'
        pool.write_text('koulu\nkirja\ntaulu\nmetsä\njärvi\n', encoding='utf-8')
        values = {'constraints': {'vocabulary_constraints': {'selection': {
            'mode': 'sample', 'pool': str(pool), 'count': 3, 'seed': 0, 'scope': 'condition',
        }}}}
        first = self.create('first', values=values)
        second = self.create('second', values=values)
        self.assertEqual(first.template['constraints'], second.template['constraints'])
        pool.unlink()
        copied = self.create('copy', copy_from='first')
        self.assertEqual(first.template['constraints'], copied.template['constraints'])

    def test_per_request_sampling_and_csv_import(self):
        csv = self.root / 'words.csv'
        csv.write_text('lemma;grade\nsähkö#virta;3\nsähkö#virta;3\nkoulu;3\nkirja;4\ntaulu;3\n', encoding='utf-8')
        spec = self.root / 'pool.json'
        write_json(spec, {
            'path': 'words.csv', 'encoding': 'utf-8', 'delimiter': ';', 'column': 'lemma',
            'where': {'grade': 3}, 'remove_compound_markers': True, 'duplicate_policy': 'unique',
        })
        self.assertEqual(read_lemmas(spec)[0], ['sähkövirta', 'koulu', 'taulu'])
        prepared = self.create(values={'constraints': {'vocabulary_constraints': {'selection': {
            'mode': 'sample', 'pool': str(spec), 'count': 2, 'seed': 0, 'scope': 'request',
        }}}})
        value = prepared.template['constraints']['vocabulary_constraints']
        self.assertEqual(value['items'], [])
        self.assertEqual(len(vocabulary_for_request(value, 5)), 2)
        self.assertEqual(vocabulary_for_request(value, 5), vocabulary_for_request(value, 5))
        self.assertGreater(len({tuple(vocabulary_for_request(value, index)) for index in range(10)}), 1)
        with self.assertRaisesRegex(ValueError, 'JSON import specification'):
            read_lemmas(csv)
