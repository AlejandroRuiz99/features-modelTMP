"""TrainModelUseCase — collect data, train, register candidate manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict

from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import ModelManifest
from HWFP.core.domain.model_status import ModelStatus
from HWFP.core.ports.model_registry import ModelRegistry
from HWFP.core.ports.model_trainer import ModelTrainer
from HWFP.core.ports.training_data_source import TrainingDataSource


@dataclass
class TrainInput:
    hyperparams: Dict[str, Any] = field(default_factory=dict)


class TrainModelUseCase:
    def __init__(
        self,
        source: TrainingDataSource,
        trainer: ModelTrainer,
        registry: ModelRegistry,
        clock: Callable[[], datetime],
        git_sha_provider: Callable[[], str],
    ) -> None:
        self._source = source
        self._trainer = trainer
        self._registry = registry
        self._clock = clock
        self._git_sha_provider = git_sha_provider

    def execute(self, inp: TrainInput) -> ModelManifest:
        # 1. Collect training examples
        examples = list(self._source.iter_examples())
        # 2. Capture dataset identity
        dataset_hash = self._source.dataset_hash()
        # 3. Train — returns opaque blob + holdout metrics
        model_blob, metrics = self._trainer.fit(examples, inp.hyperparams)
        # 4. Build candidate manifest
        trained_at = self._clock()
        git_sha = self._git_sha_provider()
        model_id = ModelId(f"model-{git_sha[:8]}-{trained_at.strftime('%Y%m%d%H%M%S')}")
        manifest = ModelManifest(
            model_id=model_id,
            trained_at=trained_at,
            git_sha=git_sha,
            dataset_hash=dataset_hash,
            dataset_rows=len(examples),
            metrics_holdout=metrics,
            gates_passed=(),
            status=ModelStatus.CANDIDATE,
        )
        # 5. Register in registry
        self._registry.register(manifest, model_blob)
        # 6. Return manifest for caller inspection
        return manifest
