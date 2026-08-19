"""ANTS test-time postprocessor.

The public class in this module remains the OpenOOD integration point. Pure
scoring, state management, label handling, and MLLM-specific behavior live in
the :mod:`openood.postprocessors.ants` package.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image

from openood.networks.clip_fixed_ood_prompt import imagenet_classes

from .ants.ens import append_feature_queue, update_adaptive_threshold
from .ants.labels import normalize_labels
from .ants.mllm import create_mllm_backend
from .ants.scoring import (
    activation_selected_score,
    find_activation_threshold,
    grouped_negative_score,
    score_with_negative_features,
)
from .ants.state import ANTSRuntimeState
from .ants.text_features import encode_text_features
from .ants.vsnl import (
    most_frequent_predictions,
    update_similar_label_cache,
)
from .ants_base_postprocessor import ANTSBasePostprocessor


class ANTSPostprocessor(ANTSBasePostprocessor):
    """Adapt CLIP's negative text space with MLLM-generated labels."""

    FAR_QUEUE_MAX_SIZE = 10_000
    MIN_ENSEMBLE_CANDIDATES = 200
    SIMILAR_CLASS_LIMIT = 40
    SIMILAR_LABEL_START_BATCH = 10
    SIMILAR_LABEL_INTERVAL = 40
    NEAR_BALANCE_REPEATS = 10

    def __init__(self, config) -> None:
        super().__init__(config)
        args = config.postprocessor.postprocessor_args

        self.tau = float(_config_value(args, 'tau', 0.0))
        self.eta = float(_config_value(args, 'eta', 0.5))
        self.in_score = str(
            _config_value(args, 'in_score', 'far_only')
        ).lower()
        self.random_permute = _as_bool(
            _config_value(args, 'random_permute', True)
        )
        self.neglabel_init_flag = _as_bool(
            _config_value(args, 'neglabel_init_flag', False)
        )
        self.save_ens_labels = _as_bool(
            _config_value(args, 'save_ens_labels', False)
        )
        self.activation_aware_ens = _as_bool(
            _config_value(args, 'activation_aware_ens', False)
        )
        self.activation_negative_count = int(
            _config_value(args, 'activation_negative_count', 1000)
        )
        self.activation_step = int(
            _config_value(args, 'activation_step', 2)
        )
        self.activation_gap = float(
            _config_value(args, 'activation_gap', 0.5)
        )
        self.activation_score_queue_size = int(
            _config_value(args, 'activation_score_queue_size', 20_000)
        )
        self.ens_stop_step = int(
            _config_value(args, 'ens_stop_step', 20)
        )
        self.similar_label_start_batch = int(
            _config_value(
                args,
                'near_label_start_batch',
                self.SIMILAR_LABEL_START_BATCH,
            )
        )
        self.similar_label_interval = int(
            _config_value(
                args,
                'near_label_interval',
                self.SIMILAR_LABEL_INTERVAL,
            )
        )
        self.mllm_model_type = str(
            _config_value(args, 'mllm_model_type', 'QWEN')
        ).upper()
        self.args_dict = _config_value(
            config.postprocessor, 'postprocessor_sweep', {}
        )

        self.class_num: Optional[int] = None
        self.group_num = 0
        self.group_len = 0
        self.reset_group_num(int(_config_value(args, 'group_num', 100)))
        self._validate_config()

        self.state = ANTSRuntimeState()
        self.mllm_backend = create_mllm_backend(self.mllm_model_type)
        self._ens_label_path: Optional[Path] = None
        self._ens_all_label_path: Optional[Path] = None
        if self.save_ens_labels:
            output_dir = Path(
                str(_config_value(config, 'output_dir', '.'))
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self._ens_all_label_path = output_dir / 'ens_labels_all.txt'
            self._ens_all_label_path.write_text('', encoding='utf-8')

    def setup(
        self,
        net: nn.Module,
        id_loader_dict: Any,
        ood_loader_dict: Any,
    ) -> None:
        """Initialize test-time memory for one ID/OOD dataset pair."""

        del id_loader_dict
        self.class_num = int(net.n_cls)
        self._prepare_ens_label_file(ood_loader_dict)
        initial_features = None
        if self.neglabel_init_flag:
            initial_features = (
                net.text_features[:, self.class_num:].t().detach()
            )
        self.state.reset(initial_features)
        self.mllm_backend.load()
        initial_queue_size = (
            0 if initial_features is None else initial_features.size(0)
        )
        print(
            '[ANTS][ENS] initialized: '
            f'updates=0/{self.ens_stop_step}, '
            f'negative_labels={initial_queue_size}, '
            f'threshold={self.state.adaptive_threshold:.4f}',
            flush=True,
        )
        if self.in_score == 'near_only':
            print(
                '[ANTS][VSNL] near-only scoring enabled: '
                f'first_update_batch={self.similar_label_start_batch}, '
                f'update_interval={self.similar_label_interval}',
                flush=True,
            )

    def reset_memory(self) -> None:
        """Clear all dynamic labels, features, counters, and histories."""

        self.state.reset()

    def reset_group_num(self, group_num: int) -> None:
        """Update the configured number of groups used for negative labels."""

        if group_num <= 0 or group_num > self.FAR_QUEUE_MAX_SIZE:
            raise ValueError(
                'group_num must be in [1, FAR_QUEUE_MAX_SIZE]'
            )
        self.group_num = group_num
        self.group_len = max(1, self.FAR_QUEUE_MAX_SIZE // group_num)

    def grouping_score(
        self, output: torch.Tensor, group_len: Optional[int] = None
    ) -> torch.Tensor:
        """Compatibility wrapper around the extracted grouped scorer."""

        self._require_setup()
        return grouped_negative_score(
            output,
            self.class_num,
            group_len or self.group_len,
            self.random_permute,
        )

    @torch.no_grad()
    def postprocess(
        self,
        net: nn.Module,
        data: Any,
        path: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict classes and produce ANTS in-distribution confidence."""

        self._require_setup()
        net.eval()
        self.state.batch_index += 1

        image_features, text_features, logit_scale = net(
            data, return_feat=True
        )
        id_text_features = text_features[:self.class_num]
        logits = logit_scale * image_features @ text_features.t()
        id_probabilities = logits[:, :self.class_num].softmax(dim=1)
        predictions = id_probabilities.argmax(dim=1)
        base_confidence = self.grouping_score(logits)

        prediction_values = predictions.detach().cpu().tolist()
        self.state.predictions.extend(prediction_values)
        self.state.all_confidences.extend(
            base_confidence.detach().cpu().unbind()
        )
        self._collect_ensemble_candidates(
            path, predictions, base_confidence
        )
        self._maybe_update_far_labels(net)
        self._maybe_update_near_labels(net)

        far_confidence = score_with_negative_features(
            image_features,
            id_text_features,
            self.state.far_negative_features,
            logit_scale,
            self.group_len,
            self.random_permute,
        )
        if self.activation_aware_ens and far_confidence is not None:
            far_confidence = self._activation_aware_far_score(
                image_features,
                id_text_features,
                logit_scale,
                far_confidence,
            )
        far_balanced = score_with_negative_features(
            image_features,
            id_text_features,
            self.state.far_negative_features,
            logit_scale,
            self.class_num,
            self.random_permute,
        )

        near_features = self.state.near_negative_features
        near_confidence = score_with_negative_features(
            image_features,
            id_text_features,
            near_features,
            logit_scale,
            self.group_len,
            self.random_permute,
        )
        repeated_near_features = (
            near_features.repeat(self.NEAR_BALANCE_REPEATS, 1)
            if near_features is not None
            else None
        )
        near_balanced = score_with_negative_features(
            image_features,
            id_text_features,
            repeated_near_features,
            logit_scale,
            self.class_num,
            self.random_permute,
        )

        confidence = self._fuse_confidences(
            base_confidence,
            far_confidence,
            far_balanced,
            near_confidence,
            near_balanced,
        )
        if torch.isnan(confidence).any():
            raise FloatingPointError('ANTS produced NaN confidence values')
        return predictions, confidence

    def _activation_aware_far_score(
        self,
        image_features: torch.Tensor,
        id_text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        reference_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Apply TANL-style transductive selection to the current ENS bank."""

        state = self.state
        state.activation_confidences.extend(
            reference_confidence.detach().cpu().unbind()
        )
        if (
            len(state.activation_confidences)
            > self.activation_score_queue_size
        ):
            del state.activation_confidences[
                :-self.activation_score_queue_size
            ]
        threshold = find_activation_threshold(
            torch.stack(state.activation_confidences)
        )

        negative_features = state.far_negative_features
        text_features = torch.cat(
            (id_text_features, negative_features), dim=0
        )
        logits = logit_scale * image_features @ text_features.t()
        score, selected_count, low_count, high_count = (
            activation_selected_score(
                logits,
                self.class_num,
                reference_confidence,
                threshold,
                self.activation_gap,
                self.activation_negative_count,
                self.activation_step,
            )
        )
        if score is None:
            return reference_confidence
        if state.batch_index == 1 or state.batch_index % 10 == 0:
            print(
                '[ANTS][TANL-ENS] activation-aware selection: '
                f'batch={state.batch_index}, '
                f'threshold={threshold:.4f}, '
                f'low={low_count}, high={high_count}, '
                f'selected={selected_count}',
                flush=True,
            )
        return score

    def set_hyperparam(self, hyperparam: Sequence[float]) -> None:
        """Set the legacy APS hyperparameter."""

        if not hyperparam:
            raise ValueError('hyperparam must contain tau')
        self.tau = float(hyperparam[0])

    def get_hyperparam(self) -> float:
        """Return the legacy APS hyperparameter."""

        return self.tau

    def get_text_features(self, net, labels):
        """Compatibility wrapper for far-label CLIP encoding."""

        return encode_text_features(net, labels)

    def get_prompt_text_features(self, net, labels):
        """Compatibility wrapper for prompted near-label CLIP encoding."""

        return encode_text_features(net, labels, 'The nice {}.')

    def _collect_ensemble_candidates(
        self,
        paths: Optional[Sequence[str]],
        predictions: torch.Tensor,
        confidences: torch.Tensor,
    ) -> None:
        if self.state.ensemble_index >= self.ens_stop_step:
            return
        if paths is None:
            return
        if len(paths) != len(confidences):
            raise ValueError('path count must match the batch size')
        for sample_path, prediction, confidence in zip(
            paths, predictions, confidences
        ):
            if float(confidence) < self.state.adaptive_threshold:
                self.state.candidate_paths.append(str(sample_path))
                self.state.candidate_predictions.append(int(prediction))

    def _maybe_update_far_labels(self, net) -> None:
        state = self.state
        if state.ensemble_index >= self.ens_stop_step:
            return
        if len(state.candidate_paths) <= self.MIN_ENSEMBLE_CANDIDATES:
            return

        threshold, upper_interval = update_adaptive_threshold(
            state.all_confidences, self.eta
        )
        state.adaptive_threshold = threshold
        state.upper_interval = upper_interval

        candidate_count = len(state.candidate_paths)
        next_update = state.ensemble_index + 1
        print(
            '[ANTS][ENS] update started: '
            f'updates={next_update}/{self.ens_stop_step}, '
            f'candidates={candidate_count}, '
            f'threshold={threshold:.4f}',
            flush=True,
        )
        labels = self._describe_candidate_images()
        self._save_ens_labels(labels)
        state.candidate_paths.clear()
        state.candidate_predictions.clear()
        text_features = encode_text_features(net, labels)
        queued_features = (
            text_features.t() if text_features is not None else None
        )
        state.far_negative_features = append_feature_queue(
            state.far_negative_features,
            queued_features,
            self.FAR_QUEUE_MAX_SIZE,
        )
        state.ensemble_index += 1
        queue_size = (
            0
            if state.far_negative_features is None
            else state.far_negative_features.size(0)
        )
        print(
            '[ANTS][ENS] update completed: '
            f'updates={state.ensemble_index}/{self.ens_stop_step}, '
            f'generated_labels={len(labels)}, '
            f'negative_labels={queue_size}',
            flush=True,
        )

    def _prepare_ens_label_file(self, ood_loader: Any) -> None:
        if not self.save_ens_labels:
            self._ens_label_path = None
            return

        dataset = getattr(ood_loader, 'dataset', None)
        dataset_name = str(getattr(dataset, 'name', 'ood'))
        safe_name = ''.join(
            character
            if character.isalnum() or character in ('-', '_')
            else '_'
            for character in dataset_name
        )
        output_dir = self._ens_all_label_path.parent
        self._ens_label_path = output_dir / f'ens_labels_{safe_name}.txt'
        self._ens_label_path.write_text('', encoding='utf-8')
        print(
            f'[ANTS][ENS] labels will be saved to {self._ens_label_path}',
            flush=True,
        )

    def _save_ens_labels(self, labels: Sequence[str]) -> None:
        if not self.save_ens_labels or not labels:
            return

        text = ''.join(f'{label}\n' for label in labels)
        for output_path in (
            self._ens_label_path,
            self._ens_all_label_path,
        ):
            if output_path is not None:
                with output_path.open('a', encoding='utf-8') as file:
                    file.write(text)
        print(
            '[ANTS][ENS] labels saved: '
            f'count={len(labels)}, file={self._ens_label_path}',
            flush=True,
        )

    def _describe_candidate_images(self):
        images = []
        for image_path in self.state.candidate_paths:
            with Image.open(image_path) as image:
                images.append(image.convert('RGB').copy())
        id_classes = [
            imagenet_classes[prediction]
            for prediction in self.state.candidate_predictions
        ]
        labels = self.mllm_backend.describe_images(images, id_classes)
        return normalize_labels(labels)

    def _maybe_update_near_labels(self, net) -> None:
        if not self.mllm_backend.supports_similar_labels:
            return
        batch_index = self.state.batch_index
        is_update_batch = (
            batch_index >= self.similar_label_start_batch
            and (
                batch_index - self.similar_label_start_batch
            ) % self.similar_label_interval == 0
        )
        if not is_update_batch:
            return

        frequent_predictions = most_frequent_predictions(
            self.state.predictions, self.SIMILAR_CLASS_LIMIT
        )
        new_predictions = [
            prediction
            for prediction in frequent_predictions
            if prediction not in self.state.similar_label_cache
        ]
        class_names = [
            imagenet_classes[prediction] for prediction in new_predictions
        ]
        print(
            '[ANTS][VSNL] update started: '
            f'batch={batch_index}, '
            f'active_classes={len(frequent_predictions)}, '
            f'new_classes={len(new_predictions)}',
            flush=True,
        )
        generated_labels = self.mllm_backend.suggest_similar_classes(
            class_names
        )
        self.state.near_negative_labels = update_similar_label_cache(
            self.state.similar_label_cache,
            frequent_predictions,
            new_predictions,
            generated_labels,
        )
        features = encode_text_features(
            net, self.state.near_negative_labels, 'The nice {}.'
        )
        self.state.near_negative_features = (
            features.t() if features is not None else None
        )
        print(
            '[ANTS][VSNL] update completed: '
            f'batch={batch_index}, '
            f'near_labels={len(self.state.near_negative_labels)}',
            flush=True,
        )

    def _fuse_confidences(
        self,
        base: torch.Tensor,
        far: Optional[torch.Tensor],
        far_balanced: Optional[torch.Tensor],
        near: Optional[torch.Tensor],
        near_balanced: Optional[torch.Tensor],
    ) -> torch.Tensor:
        far = base if far is None else far
        far_balanced = base if far_balanced is None else far_balanced
        near = base if near is None else near
        near_balanced = base if near_balanced is None else near_balanced

        if self.in_score == 'far_only':
            return far
        if self.in_score == 'near_only':
            return near
        if self.in_score != 'ada':
            raise ValueError(
                'in_score must be one of: ada, far_only, near_only'
            )

        self.state.far_confidences.extend(
            far_balanced.detach().cpu().unbind()
        )
        self.state.near_confidences.extend(
            near_balanced.detach().cpu().unbind()
        )
        far_mean = torch.stack(self.state.far_confidences).mean()
        near_mean = torch.stack(self.state.near_confidences).mean()
        far_uncertainty = 1.0 - far_mean
        near_uncertainty = 1.0 - near_mean
        denominator = far_uncertainty + near_uncertainty
        if float(denominator.abs()) < torch.finfo(denominator.dtype).eps:
            weight = 0.5
        else:
            weight = float(far_uncertainty / denominator)
        return weight * far + (1.0 - weight) * near

    def _validate_config(self) -> None:
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError('eta must be between zero and one')
        if self.ens_stop_step < 0:
            raise ValueError('ens_stop_step must be non-negative')
        if self.activation_negative_count <= 0:
            raise ValueError('activation_negative_count must be positive')
        if self.activation_step <= 0:
            raise ValueError('activation_step must be positive')
        if not 0.0 <= self.activation_gap <= 1.0:
            raise ValueError('activation_gap must be between zero and one')
        if self.activation_score_queue_size <= 0:
            raise ValueError(
                'activation_score_queue_size must be positive'
            )
        if self.similar_label_start_batch <= 0:
            raise ValueError('near_label_start_batch must be positive')
        if self.similar_label_interval <= 0:
            raise ValueError('near_label_interval must be positive')
        if self.in_score not in ('ada', 'far_only', 'near_only'):
            raise ValueError(
                'in_score must be one of: ada, far_only, near_only'
            )

    def _require_setup(self) -> None:
        if self.class_num is None:
            raise RuntimeError('setup() must be called before postprocess()')


# Historical code imported this spelling. Keep it as a compatibility alias.
ANTSprocessor = ANTSPostprocessor


def _config_value(config, key: str, default):
    value = getattr(config, key, None)
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', '1', 'yes', 'on'):
            return True
        if normalized in ('false', '0', 'no', 'off'):
            return False
    raise ValueError(f'expected a boolean value, got {value!r}')
