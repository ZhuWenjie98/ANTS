"""Lightweight tests for ANTS modules that do not load large models."""

import unittest

import torch

from openood.postprocessors.ants.ens import append_feature_queue
from openood.postprocessors.ants.labels import (
    normalize_labels,
    parse_numbered_labels,
)
from openood.postprocessors.ants.mllm.factory import create_mllm_backend
from openood.postprocessors.ants.scoring import (
    activation_selected_score,
    cumulative_negative_score,
    find_activation_threshold,
    grouped_negative_score,
    positive_probability_score,
)
from openood.postprocessors.ants.state import ANTSRuntimeState
from openood.postprocessors.ants.vsnl import update_similar_label_cache


class ScoringTest(unittest.TestCase):
    def test_grouped_score_has_batch_shape_and_preserves_rng(self):
        logits = torch.tensor([
            [3.0, 1.0, 0.5, -0.5, 0.0, 1.0],
            [0.0, 2.0, 1.0, 0.5, -1.0, 0.0],
        ])
        rng_before = torch.random.get_rng_state()

        scores = grouped_negative_score(
            logits,
            num_id_classes=2,
            group_size=2,
            random_permute=True,
        )

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(torch.all((scores >= 0.0) & (scores <= 1.0)))

    def test_probability_score_rejects_invalid_class_count(self):
        with self.assertRaises(ValueError):
            positive_probability_score(torch.ones(2, 3), 4)

    def test_cumulative_negative_score_matches_prefix_softmax(self):
        logits = torch.tensor([[2.0, 1.0, 0.5, -0.5, 0.25]])
        expected = torch.stack([
            logits[:, :4].softmax(dim=1)[:, :2].sum(dim=1),
            logits.softmax(dim=1)[:, :2].sum(dim=1),
        ]).mean(dim=0)

        result = cumulative_negative_score(
            logits, num_id_classes=2, step=2
        )

        self.assertTrue(torch.allclose(result, expected))

    def test_activation_selection_prefers_low_confidence_response(self):
        logits = torch.tensor([
            [2.0, 0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0, 0.0],
        ])
        confidence = torch.tensor([0.1, 0.9])

        score, selected, low_count, high_count = activation_selected_score(
            logits,
            num_id_classes=2,
            reference_confidence=confidence,
            threshold=0.5,
            gap=0.0,
            max_negatives=1,
            step=2,
        )

        expected = positive_probability_score(logits[:, [0, 1, 3]], 2)
        self.assertTrue(torch.allclose(score, expected))
        self.assertEqual((selected, low_count, high_count), (1, 1, 1))

    def test_activation_threshold_separates_two_score_modes(self):
        scores = torch.tensor([0.1, 0.12, 0.15, 0.8, 0.85, 0.9])
        threshold = find_activation_threshold(scores)
        self.assertGreater(threshold, 0.15)
        self.assertLessEqual(threshold, 0.8)


class StateAndLabelTest(unittest.TestCase):
    def test_state_reset_clears_dynamic_memory(self):
        state = ANTSRuntimeState(
            batch_index=3,
            candidate_paths=['image.jpg'],
            predictions=[1, 2],
            activation_confidences=[torch.tensor(0.5)],
            far_negative_features=torch.ones(2, 4),
        )
        initial_features = torch.zeros(1, 4)

        state.reset(initial_features)

        self.assertEqual(state.batch_index, 0)
        self.assertEqual(state.candidate_paths, [])
        self.assertEqual(state.predictions, [])
        self.assertEqual(state.activation_confidences, [])
        self.assertIs(state.far_negative_features, initial_features)

    def test_label_parsing_is_ordered_and_deduplicated(self):
        parsed = parse_numbered_labels(
            '1. tabby cat\n2) red fox\n3. Tabby   Cat.\nnot a label'
        )
        self.assertEqual(parsed, ['tabby cat', 'red fox'])
        self.assertEqual(
            normalize_labels([' dog. ', '', 'Dog', 'red  fox']),
            ['dog', 'red fox'],
        )

    def test_similar_label_cache_drops_inactive_predictions(self):
        cache = {1: ['old label'], 2: ['inactive']}
        labels = update_similar_label_cache(
            cache,
            active_predictions=[1, 3],
            generated_predictions=[3],
            generated_labels=[['new label', 'New label']],
        )
        self.assertEqual(cache, {1: ['old label'], 3: ['new label']})
        self.assertEqual(labels, ['old label', 'new label'])

    def test_feature_queue_is_bounded(self):
        queue = torch.tensor([[1.0], [2.0]])
        new_features = torch.tensor([[3.0], [4.0]])
        result = append_feature_queue(queue, new_features, max_size=3)
        self.assertEqual(result[:, 0].tolist(), [4.0, 1.0, 2.0])

class BackendFactoryTest(unittest.TestCase):
    def test_factory_normalizes_names_without_loading_models(self):
        self.assertEqual(
            create_mllm_backend('qwen').__class__.__name__, 'QwenBackend'
        )
        self.assertEqual(
            create_mllm_backend('blip2').__class__.__name__, 'BlipBackend'
        )

    def test_factory_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            create_mllm_backend('unknown')


if __name__ == '__main__':
    unittest.main()
