"""Smoke test for the public OpenOOD ANTS registration."""

import unittest

import torch
import torch.nn as nn

from openood.postprocessors.ants_postprocessor import ANTSPostprocessor
from openood.postprocessors.utils import get_postprocessor
from openood.utils import Config


def _ants_config(**overrides):
    args = {
        'eta': 0.5,
        'group_num': 100,
        'random_permute': True,
        'in_score': 'far_only',
        'neglabel_init_flag': False,
        'ens_stop_step': 20,
        'mllm_model_type': 'QWEN',
    }
    args.update(overrides)
    return Config({
        'postprocessor': {
            'name': 'ants',
            'postprocessor_args': args,
        },
    })


class _FakeBackend:
    supports_similar_labels = False

    def load(self):
        pass


class _FakeNetwork(nn.Module):
    n_cls = 2

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.text_features = torch.tensor([
            [1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, 0.5, 0.5],
        ])

    def forward(self, data, return_feat=False):
        del data
        image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        if return_feat:
            return image_features, self.text_features.t(), torch.tensor(1.0)
        return image_features @ self.text_features


class ANTSRegistrationTest(unittest.TestCase):
    def test_ants_name_constructs_public_postprocessor(self):
        postprocessor = get_postprocessor(_ants_config())

        self.assertIsInstance(postprocessor, ANTSPostprocessor)
        self.assertEqual(postprocessor.group_len, 100)

    def test_two_argument_postprocess_skips_path_dependent_collection(self):
        postprocessor = ANTSPostprocessor(_ants_config())
        postprocessor.mllm_backend = _FakeBackend()
        network = _FakeNetwork()
        postprocessor.setup(network, None, None)

        predictions, confidences = postprocessor.postprocess(
            network, torch.zeros(2, 1)
        )

        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertEqual(tuple(confidences.shape), (2,))
        self.assertEqual(postprocessor.state.candidate_paths, [])

    def test_invalid_score_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'in_score'):
            ANTSPostprocessor(_ants_config(in_score='sum'))


if __name__ == '__main__':
    unittest.main()
