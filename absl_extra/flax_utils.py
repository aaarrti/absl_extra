from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    no_type_check,
)

import clu.periodic_actions
import jax
import jax.numpy as jnp
from absl import logging
from clu.metrics import Metric
from flax import jax_utils
from flax import struct
from flax.core import frozen_dict
from flax.training import early_stopping, train_state, common_utils
from jaxtyping import Array, Float, Int, Key, jaxtyped, Int32

from absl_extra.jax_utils import prefetch_to_device

T = TypeVar("T", contravariant=True)
TS = TypeVar("TS", bound=train_state.TrainState, contravariant=True)
M = TypeVar("M", bound=clu.metrics.Collection, contravariant=True)
DatasetFactory = Callable[[], Iterable[Tuple[T, Int[Array, "batch classes"]]]]  # noqa


@struct.dataclass
class NanSafeAverage(clu.metrics.Average):
    def compute(self) -> float:
        if self.count != 0:
            return super().compute()
        else:
            return 0


@struct.dataclass
class F1Score(clu.metrics.Metric):
    """
    Class F1Score
    This class represents the F1 Score metric for evaluating classification models.

    - A model will obtain a high F1 score if both Precision and Recall are high.
    - A model will obtain a low F1 score if both Precision and Recall are low.
    - A model will obtain a medium F1 score if one of Precision and Recall is low and the other is high.
    - Precision: Precision is a measure of how many of the positively classified examples were actually positive.
    - Recall (also called Sensitivity or True Positive Rate): Recall is a measure of how many of the actual positive
    examples were correctly labeled by the classifier.

    """

    true_positive: Float[Array, "1"]
    false_positive: Float[Array, "1"]
    false_negative: Float[Array, "1"]

    @classmethod
    def from_model_output(
        cls,
        *,
        logits: Float[Array, "batch classes"],
        labels: Int32[Array, "batch classes"],
        threshold: float = 0.5,
        **kwargs,
    ) -> "F1Score":
        predicted = jnp.asarray(logits >= threshold, labels.dtype)
        true_positive = jnp.sum((predicted == 1) & (labels == 1))
        false_positive = jnp.sum((predicted == 1) & (labels == 0))
        false_negative = jnp.sum((predicted == 0) & (labels == 1))

        return F1Score(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    def merge(self, other: "F1Score") -> "F1Score":
        return F1Score(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
        )

    @classmethod
    def empty(cls) -> "F1Score":
        return F1Score(
            true_positive=0,
            false_positive=0,
            false_negative=0,
        )

    def compute(self) -> float:
        precision = nan_div(
            self.true_positive, self.true_positive + self.false_positive
        )
        recall = nan_div(self.true_positive, self.true_positive + self.false_negative)

        # Ensure we don't divide by zero if both precision and recall are zero
        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


@struct.dataclass
class BinaryAccuracy(NanSafeAverage):
    @classmethod
    def from_model_output(  # noqa
        cls,
        *,
        logits: Float[Array, "batch classes"],
        labels: Int32[Array, "batch classes"],
        threshold: float = 0.5,
        **kwargs,
    ) -> "BinaryAccuracy":
        predicted = jnp.asarray(logits >= threshold, logits.dtype)
        return super().from_model_output(
            values=jnp.asarray(predicted == labels, predicted.dtype)
        )


class ApplyFn(Protocol[T]):
    def __call__(
        self,
        params: Dict[str, frozen_dict.FrozenDict],
        x_batch: T,
        train: bool = False,
        rngs: Dict[str, Key] | None = None,
        **kwargs,
    ) -> Float[Array, "batch classes"]:  # noqa
        ...


class TrainingHook(Protocol[TS, M]):
    def __call__(
        self,
        step: int,
        *,
        training_state: TS,
        training_metrics: M,
        **kwargs,
    ) -> None:
        ...


class ValidationHook(Protocol[M]):
    def __call__(
        self,
        step: int,
        *,
        validation_metrics: M,
        **kwargs,
    ) -> None:
        ...


class UncheckedReportProgress(clu.periodic_actions.ReportProgress):
    def __call__(self, step: int, **kwargs) -> bool:
        return super().__call__(int(step))

    @no_type_check
    def _init_and_check(self, step: int, t: float):
        """Initializes and checks it was called at every step."""
        if self._previous_step is None:
            self._previous_step = step
            self._previous_time = t
            self._last_step = step
        else:
            self._last_step = step


class UncheckedPeriodicCallback(clu.periodic_actions.PeriodicCallback):
    def __call__(self, step: int, *args, **kwargs) -> bool:
        return super().__call__(int(step), *args, **kwargs)

    @no_type_check
    def _init_and_check(self, step: int, t: float):
        """Initializes and checks it was called at every step."""
        if self._previous_step is None:
            self._previous_step = step
            self._previous_time = t
            self._last_step = step
        else:
            self._last_step = step


def save_as_msgpack(
    params: frozen_dict.FrozenDict, save_path: str = "model.msgpack"
) -> None:
    """
    Parameters
    ----------
    params : frozen_dict.FrozenDict
        The frozen dictionary object that contains the parameters to be saved.
    save_path : str, optional
        The file path where the msgpack file will be saved. Default is "model.msgpack".

    Returns
    -------
    None
        This method does not return any value.
    """
    logging.info(f"Writing {save_path}")
    msgpack_bytes: bytes = frozen_dict.serialization.to_bytes(params)
    with open(save_path, "wb+") as file:
        file.write(msgpack_bytes)


def load_from_msgpack(
    params: frozen_dict.FrozenDict, save_path: str = "model.msgpack"
) -> frozen_dict.FrozenDict:
    """
    Load model parameters from a msgpack file.

    Parameters
    ----------
    params : frozen_dict.FrozenDict
        The original parameters of the model.
    save_path : str, optional
        The path to the msgpack file containing the serialized parameters.
        Default is "model.msgpack".

    Returns
    -------
    params : frozen_dict.FrozenDict
        The loaded parameters.

    """
    logging.info(f"Reading {save_path}")

    with open(save_path, "rb") as file:
        bytes_data = file.read()

    params = frozen_dict.serialization.from_bytes(params, bytes_data)

    return params


class InvalidEpochsNumberError(RuntimeError):
    def __init__(self, value: int):
        super().__init__(f"Epochs must be greater than 0, but found {value}")


@jaxtyped
def fit(
    *,
    training_state: TS,
    training_dataset_factory: DatasetFactory,
    validation_dataset_factory: DatasetFactory,
    metrics_container_type: Type[M],
    # fmt: off
    training_step_func: Callable[[TS, T, Int[Array, "batch classes"], M], Tuple[TS, M]],  # noqa
    validation_step_func: Callable[[TS, T, Int[Array, "batch classes"], M], M],
    # noqa
    # fmt: on
    training_hooks: List[TrainingHook[TS, M]] | None = None,
    validation_hooks: List[ValidationHook[M]] | None = None,
    epochs: int = 1,
    prefetch_buffer_size: int = 2,
    verbose: bool = False,
) -> Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]:
    """
    Parameters
    ----------
    training_state : TS
        The initial state of the training process.
    training_dataset_factory : DatasetFactory
        A factory function that returns the training dataset.
    validation_dataset_factory : DatasetFactory
        A factory function that returns the validation dataset.
    metrics_container_type : Type[M]
        The type of container to store the metrics.
    training_step_func : Callable[[TS, T, Int[Array, "batch classes"]], Tuple[TS, M]]
        A function that performs a single training step. It takes the training state, input data, and target data as inputs,
        and returns the updated training state and metrics.
    validation_step_func : Callable[[TS, T, Int[Array, "batch classes"]], M]
        A function that performs a single validation step. It takes the training state, input data, and target data as inputs,
        and returns the metrics.
    training_hooks : List[TrainingHook[TS, M]] | None, optional
        A list of training hooks to be executed after each training step. Defaults to None.
    validation_hooks : List[ValidationHook[M]] | None, optional
        A list of validation hooks to be executed after each validation step. Defaults to None.
    epochs : int, optional
        The number of training epochs. Defaults to 1.
    prefetch_buffer_size : int, optional
        The size of the prefetch buffer for loading data. Defaults to 2.
    verbose : bool, optional
        Whether to display verbose output during training. Defaults to False.

    Returns
    -------
    Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]
        A tuple containing the training and validation metrics, and the final training state parameters.
    """
    if epochs <= 0:
        raise InvalidEpochsNumberError(epochs)

    is_multi_device = len(jax.devices()) > 1

    if is_multi_device:
        dropout_key = common_utils.shard_prng_key(training_state.dropout_key)
        training_state = jax_utils.replicate(training_state)
        training_state.replace(dropout_key=dropout_key)
    
    def step_num():
        step = training_state.step
        if is_multi_device:
            step = jax_utils.unreplicate(step)
        return int(step)
        
    if training_hooks is None:
        training_hooks = []
    if validation_hooks is None:
        validation_hooks = []

    for epoch in range(epochs):
        if verbose:
            logging.info(f"Epoch {epoch + 1}/{epochs}...")

        training_dataset = training_dataset_factory()
        if prefetch_buffer_size != 0:
            if is_multi_device:
                training_dataset = jax_utils.prefetch_to_device(
                    training_dataset, prefetch_buffer_size
                )
            else:
                training_dataset = prefetch_to_device(
                    training_dataset, prefetch_buffer_size
                )

        training_metrics = metrics_container_type.empty()

        if is_multi_device:
            training_metrics = jax_utils.replicate(training_metrics)

        for x_batch, y_batch in training_dataset:
            training_state, training_metrics = training_step_func(
                training_state, x_batch, y_batch, training_metrics
            )

            for hook in training_hooks:
                hook(
                    step_num(),
                    training_state=training_state,
                    training_metrics=training_metrics.unreplicate(),
                )
        if verbose:
            logging.info(
                {f"train_{k}": f"{float(v):.3f}"}
                for k, v in training_metrics.unreplicate().compute().items()
            )

        validation_dataset = validation_dataset_factory()
        if prefetch_buffer_size != 0:
            if is_multi_device:
                validation_dataset = jax_utils.prefetch_to_device(
                    validation_dataset, prefetch_buffer_size
                )
            else:
                validation_dataset = prefetch_to_device(
                    validation_dataset, prefetch_buffer_size
                )
        validation_metrics = metrics_container_type.empty()

        if is_multi_device:
            validation_metrics = jax_utils.replicate(validation_metrics)

        for x_batch, y_batch in validation_dataset:
            validation_metrics = validation_step_func(
                training_state, x_batch, y_batch, validation_metrics
            )

        if verbose:
            logging.info(
                {f"val_{k}": f"{float(v):.3f}"}
                for k, v in validation_metrics.unreplicate().compute().items()
            )

        for val_hook in validation_hooks:
            val_hook(
                step_num(),
                validation_metrics=validation_metrics.unreplicate(),
            )

            if isinstance(val_hook, early_stopping.EarlyStopping):
                if val_hook.should_stop:
                    break

    if is_multi_device:
        training_state = jax_utils.unreplicate(training_state)

    return (
        training_metrics.unreplicate().compute(),  # noqa
        validation_metrics.unreplicate().compute(),  # noqa
    ), training_state.params


def nan_div(a: float, b: float) -> float:
    if b == 0:
        return 0
    else:
        return a / b
