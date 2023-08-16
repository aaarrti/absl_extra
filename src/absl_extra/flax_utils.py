from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Iterable,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    no_type_check,
    runtime_checkable,
    Sequence,
    List,
    Any,
)
import dataclasses
import clu.metrics
import clu.periodic_actions
import clu.metric_writers
import jax
from jax.sharding import NamedSharding
import jax.numpy as jnp
from absl import logging
from flax import jax_utils, struct
from flax.core import frozen_dict
from flax.training import common_utils, train_state
from jaxtyping import Array, Float, Int, Int32, jaxtyped

from absl_extra.jax_utils import prefetch_to_device
from absl_extra.keras_pbar import keras_pbar
from absl_extra.dataclass import dataclass

T = TypeVar("T")
TS = TypeVar("TS", bound=train_state.TrainState)
M = TypeVar("M", bound=clu.metrics.Collection)
S = TypeVar("S", bound=Sequence)
DatasetFactory = Callable[[], Iterable[Tuple[T, Int[Array, "batch classes"]]]]  # noqa
ValidationStep = Callable[[TS, T, Int[Array, "batch classes"]], Tuple[TS, M]]  # noqa
TrainingStep = Callable[[TS, T, Int[Array, "batch classes"]], Tuple[TS, M]]  # noqa
MetricsAndParams = Tuple[
    Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict
]


@runtime_checkable
class EarlyStopping(Protocol):
    should_stop: bool


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
        logits: Float[Array, "batch classes"],  # noqa
        labels: Int32[Array, "batch classes"],  # noqa
        threshold: float = 0.5,
        **kwargs,
    ) -> "F1Score":
        probs = jax.nn.sigmoid(logits)
        predicted = jnp.asarray(probs >= threshold, labels.dtype)
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
        precision = _nan_div(
            self.true_positive, self.true_positive + self.false_positive
        )
        recall = _nan_div(self.true_positive, self.true_positive + self.false_negative)

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
        logits: Float[Array, "batch classes"],  # noqa
        labels: Int32[Array, "batch classes"],  # noqa
        threshold: float = 0.5,
        **kwargs,
    ) -> "BinaryAccuracy":
        predicted = jnp.asarray(logits >= threshold, logits.dtype)
        return super().from_model_output(
            values=jnp.asarray(predicted == labels, predicted.dtype)
        )


class OnStepBegin(Protocol[TS, M]):  # type: ignore
    def __call__(self, step: int) -> None:
        ...


class OnStepEnd(Protocol[TS, M]):  # type: ignore
    def __call__(self, step: int, *, training_metrics: M, training_state: TS) -> None:  # type: ignore
        ...


class OnEpochBegin(Protocol[TS, M]):  # type: ignore
    def __call__(self, step: int) -> None:
        ...


class OnEpochEnd(Protocol[TS, M]):  # type: ignore
    def __call__(self, step: int, *, validation_metrics: M, training_state: TS) -> None:  # type: ignore
        ...


class OnTrainingBegin(Protocol[TS, M]):  # type: ignore
    def __call__(
        self,
        step: int,
        *,
        training_metrics: M,  # type: ignore
        validation_metrics: M,  # type: ignore
        training_state: TS,  # type: ignore
    ) -> None:  # type: ignore
        ...


class OnTrainingEnd(Protocol[TS, M]):  # type: ignore
    def __call__(
        self,
        step: int,
        *,
        training_metrics: M,  # type: ignore
        validation_metrics: M,  # type: ignore
        training_state: TS,  # type: ignore
    ) -> None:  # type: ignore
        ...


@dataclass
class TrainingHooks:
    on_epoch_begin: List[OnEpochBegin] = dataclasses.field(default_factory=list)
    on_epoch_end: List[OnEpochEnd] = dataclasses.field(default_factory=list)
    on_step_begin: List[OnStepBegin] = dataclasses.field(default_factory=list)
    on_step_end: List[OnStepEnd] = dataclasses.field(default_factory=list)
    on_training_begin: List[OnTrainingBegin] = dataclasses.field(default_factory=list)
    on_training_end: List[OnTrainingEnd] = dataclasses.field(default_factory=list)


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


class InvalidEpochsNumberError(RuntimeError):
    def __init__(self, value: int):
        super().__init__(f"Epochs must be greater than 0, but found {value}")


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


@jaxtyped
def fit_single_device(
    *,
    training_state: TS,  # type: ignore
    metrics_container_type: Type[M],
    training_step_func: TrainingStep,
    training_dataset_factory: DatasetFactory,
    validation_dataset_factory: DatasetFactory,
    validation_step_func: ValidationStep,
    
    epochs: int = 1,
    prefetch_buffer_size: int = 2,
    verbose: bool = True,
    hooks: TrainingHooks | None = None,
    num_training_steps: int | None = None,
) -> MetricsAndParams:
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
    hooks : List[TrainingHook[TS, M]] | None, optional
        A list of training hooks to be executed before and after each training step. Defaults to None.
    epochs : int, optional
        The number of training epochs. Defaults to 1.
    prefetch_buffer_size : int, optional
        The size of the prefetch buffer for loading data. Defaults to 2. Set to 0 for TPU.
    verbose : bool, optional
        Whether to display verbose output during training. Defaults to False.
    num_training_steps:
        Must be provided in cases verbose=True, and dataset is not typing.Sized.
    Returns
    -------
    Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]
        A tuple containing the training and validation metrics, and the final training state parameters.
    """
    
    if epochs <= 0:
        raise InvalidEpochsNumberError(epochs)

    if hooks is None:
        hooks = TrainingHooks()
        
    for epoch in range(epochs):
        if verbose:
            logging.info(f"Epoch {epoch + 1}/{epochs}...")

        for hook in hooks.on_epoch_begin:
            hook(int(training_state.step))

        training_dataset = training_dataset_factory()

        if prefetch_buffer_size != 0:
            prefetch_to_device(training_dataset, prefetch_buffer_size)

        if verbose:
            training_dataset = keras_pbar(training_dataset, n=num_training_steps)
        training_metrics = metrics_container_type.empty()

        for x_batch, y_batch in training_dataset:
            for hook in hooks.on_step_begin:
                hook(int(training_state.step))

            training_state, training_metrics_i = training_step_func(
                training_state, x_batch, y_batch
            )
            training_metrics = training_metrics.merge(training_metrics_i)

            for hook in hooks.on_step_end:  # type: ignore
                hook(
                    int(training_state.step),
                    training_metrics=training_metrics,  # type: ignore
                    training_state=training_state,  # type: ignore
                )
                if isinstance(hook, EarlyStopping) and hook.should_stop:
                    break
        if verbose:
            logging.info(
                {f"train_{k}": f"{float(v):.3f}"}
                for k, v in training_metrics.compute().items()
            )

        validation_dataset = validation_dataset_factory()
        if prefetch_buffer_size != 0:
            validation_dataset = prefetch_to_device(
                validation_dataset, prefetch_buffer_size
            )
        validation_metrics = metrics_container_type.empty()

        for x_batch, y_batch in validation_dataset:
            validation_metrics_i = validation_step_func(
                training_state, x_batch, y_batch
            )
            validation_metrics = validation_metrics.merge(validation_metrics_i)

        if verbose:
            logging.info(
                {f"val_{k}": f"{float(v):.3f}"}
                for k, v in validation_metrics.compute().items()
            )

        for hook in hooks.on_epoch_end:  # type: ignore
            hook(  # type: ignore
                int(training_state.step),
                training_state=training_state,
                validation_metrics=validation_metrics,
            )
            if isinstance(hook, EarlyStopping) and hook.should_stop:
                break

    params = training_state.params
    training_metrics = training_metrics.compute()  # noqa
    validation_metrics = validation_metrics.compute()  # noqa

    return (training_metrics, validation_metrics), params


def fit_multi_device(
    *,
    training_state: TS,  # type: ignore
    metrics_container_type: Type[M],
    training_step_func: TrainingStep,
    training_dataset_factory: DatasetFactory,
    validation_dataset_factory: DatasetFactory,
    validation_step_func: ValidationStep,
    hooks: TrainingHooks | None = None,
    epochs: int = 1,
    prefetch_buffer_size: int = 2,
    verbose: bool = False,
    num_training_steps: int | None = None,
    skip_shard: bool = False,
    data_sharding: NamedSharding | None = None,
    params_replication: NamedSharding | None = None,
) -> MetricsAndParams:
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
    hooks : List[TrainingHook[TS, M]] | None, optional
        A list of training hooks to be executed before and after each training step. Defaults to None.
    epochs : int, optional
        The number of training epochs. Defaults to 1.
    prefetch_buffer_size : int, optional
        The size of the prefetch buffer for loading data. Defaults to 2. Set to 0 for TPU.
    verbose : bool, optional
        Whether to display verbose output during training. Defaults to False.
    num_training_steps:
        Must be provided in cases verbose=True, and dataset is not typing.Sized.
    skip_shard:
        If set to True, will skip sharding of data before passing it to training_step_func
        and validation_step_func. You might want it, in case your train step is decorated
        with @pad_shard_unpad. Applies only to distributed training.
    data_sharding:
        NamesSharding, in case you want more fine-grained control on how data is sharded across devices.
        Applies only to distributed training.
    params_replication:
        NamedSharding, in case you want more fine-grained control on how params are replicated across replicas,
        e.g., you might want to shard large kernel instead of replicating them (or both).

    Returns
    -------
    Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]
        A tuple containing the training and validation metrics, and the final training state parameters.
    """
    
    if epochs <= 0:
        raise InvalidEpochsNumberError(epochs)

    if hooks is None:
        hooks = TrainingHooks()
    
    # How do we handle batch stats?
    training_state = jax_utils.replicate(training_state)
    if hasattr(training_state, "dropout_key"):
        training_state.replace(
            dropout_key=common_utils.shard_prng_key(
                jax_utils.unreplicate(training_state.dropout_key)
            )
        )

    def step_number():
        return int(jax_utils.unreplicate(training_state.step))

    def shard_x_y(x, y):
        if not skip_shard:
            if data_sharding is not None:
                x = jax.device_put(x, data_sharding)
                y = jax.device_put(y, data_sharding)
            else:
                x = common_utils.shard(x)
                y = common_utils.shard(y)

        return x, y

    for epoch in range(epochs):
        if verbose:
            logging.info(f"Epoch {epoch + 1}/{epochs}...")

        for hook in hooks.on_epoch_begin:
            hook(step_number())

        training_dataset = training_dataset_factory()
        if prefetch_buffer_size != 0:
            training_dataset = jax_utils.prefetch_to_device(
                training_dataset, prefetch_buffer_size
            )

        if verbose:
            training_dataset = keras_pbar(training_dataset, n=num_training_steps)

        training_metrics = jax_utils.replicate(metrics_container_type.empty())

        for x_batch, y_batch in training_dataset:
            x_batch, y_batch = shard_x_y(x_batch, y_batch)

            for hook in hooks.on_step_begin:
                hook(step_number())

            training_state, training_metrics_i = training_step_func(
                training_state, x_batch, y_batch
            )

            training_metrics = training_metrics.merge(training_metrics_i)

            for hook in hooks.on_step_end:  # type: ignore
                hook(  # type: ignore
                    step_number(),
                    training_metrics=training_metrics.unreplicate(),
                    training_state=jax_utils.unreplicate(training_state),
                )
                if isinstance(hook, EarlyStopping) and hook.should_stop:
                    break
        if verbose:
            logging.info(
                {f"train_{k}": f"{float(v):.3f}"}
                for k, v in training_metrics.unreplicate().compute().items()
            )

        validation_dataset = validation_dataset_factory()
        if prefetch_buffer_size != 0:
            validation_dataset = jax_utils.prefetch_to_device(
                validation_dataset, prefetch_buffer_size
            )

        validation_metrics = jax_utils.replicate(metrics_container_type.empty())

        for x_batch, y_batch in validation_dataset:
            x_batch, y_batch = shard_x_y(x_batch, y_batch)
            validation_metrics_i = validation_step_func(
                training_state, x_batch, y_batch
            )
            validation_metrics = validation_metrics.merge(validation_metrics_i)

        if verbose:
            logging.info(
                {f"val_{k}": f"{float(v):.3f}"}
                for k, v in validation_metrics.unreplicate().compute().items()
            )

        for hook in hooks.on_epoch_end:  # type: ignore
            hook(  # type: ignore
                step_number(),
                training_state=jax_utils.unreplicate(training_state),
                validation_metrics=validation_metrics.unreplicate(),
            )
            if isinstance(hook, EarlyStopping) and hook.should_stop:
                break

    params = jax_utils.unreplicate(training_state).params
    training_metrics = training_metrics.unreplicate().compute()  # noqa
    validation_metrics = validation_metrics.unreplicate().compute()  # noqa

    return (training_metrics, validation_metrics), params


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    log_frequency: int,
    logdir: str = "tensorboard",
    hyperparams_factory: Callable[[], Dict[str, Any]] | None = None,
) -> TrainingHooks:
    """
    Create typical training hooks

    - training metrics writer
    - validation metrics writer
    - report progress

    Parameters
    ----------
    num_training_steps
    epochs
    log_frequency:
        Number of times per epoch to write metrics/report progress.
    hyperparams_factory:
        If not None, will write return value as hyperparams at beginning of each epoch to
        use with Tensorboard visualization.
    logdir:
        Directory where to write metrics into.

    Returns
    -------

    """

    training_writer = clu.metric_writers.create_default_writer(
        logdir=logdir, collection="training"
    )
    validation_writer = clu.metric_writers.create_default_writer(
        logdir=logdir, collection="validation"
    )

    report_progress = clu.periodic_actions.ReportProgress(
        on_steps=[1, num_training_steps * epochs],
        every_steps=num_training_steps // log_frequency,
        num_train_steps=num_training_steps * epochs,
        writer=training_writer,
    )

    def _flush(*args, **kwargs):
        training_writer.flush()
        validation_writer.flush()

    on_train_begin = []

    if hyperparams_factory is not None:
        on_train_begin.append(
            lambda *args, **kwargs: training_writer.write_hparams(hyperparams_factory())  # type: ignore
        )

    on_step_begin = [
        lambda step, *args, **kwargs: report_progress(step),
    ]

    on_step_end = [
        clu.periodic_actions.PeriodicCallback(
            on_steps=[1, num_training_steps * epochs],
            every_steps=num_training_steps // log_frequency,
            callback_fn=lambda step, *args, training_metrics, **kwargs: training_writer.write_scalars(
                step, training_metrics.compute()
            ),
        ),
    ]

    on_epoch_end = [
        UncheckedPeriodicCallback(
            on_steps=[1, num_training_steps * epochs],
            every_steps=num_training_steps // log_frequency,
            callback_fn=lambda step, *args, validation_metrics, **kwargs: validation_writer.write_scalars(
                step, validation_metrics.compute()
            ),
        ),
    ]

    on_train_end = [lambda *args, **kwargs: _flush]

    return TrainingHooks(  # type: ignore
        on_training_begin=on_train_begin,
        on_step_begin=on_step_begin,
        on_epoch_end=on_epoch_end,
        on_step_end=on_step_end,
        on_training_end=on_train_end,
    )


def _nan_div(a: float, b: float) -> float:
    if b == 0:
        return 0
    else:
        return a / b
