from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    ContextManager,
    Literal,
    Optional,
    Union,
    Mapping
)

import clu.metric_writers
import clu.metrics
import clu.periodic_actions
import tensorflow as tf
from absl import logging
from clu.metrics import Collection
from flax import jax_utils
from flax.core import frozen_dict
from flax.struct import dataclass
from flax.training import common_utils, train_state
from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Int, jaxtyped, PyTree
from tqdm.auto import tqdm

from absl_extra.clu_utils import UncheckedPeriodicCallback
from absl_extra.jax_utils import prefetch_to_device
from absl_extra.logging_utils import log_exception
from absl_extra.typing_utils import ParamSpec

P = ParamSpec("P")
T = TypeVar("T", bound=Union[Array, PyTree])
TS = TypeVar("TS", bound=train_state.TrainState)
S = TypeVar("S", bound=Sequence)
DatasetFactory = Callable[[], Iterable[Tuple[T, Array]]]


class InvalidEpochsNumberError(RuntimeError):
    def __init__(self, value: int):
        super().__init__(f"Epochs must be greater than 0, but found {value}")


M = TypeVar("M", bound=Collection)
ValidationStep = Callable[[TS, T, Array], Tuple[TS, M]]
TrainingStep = Callable[[TS, T, Array], Tuple[TS, M]]
MetricsAndParams = Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]
StepType = Literal["training", "validation"]


class OnStepEnd(Protocol[TS, M]):
    def __call__(self, step: int, *, training_metrics: M, training_state: TS) -> Mapping[str, M | TS] | None:
        ...


class OnEpochEnd(Protocol[TS, M]):
    def __call__(self, epoch: int, *, validation_metrics: M, training_state: TS) -> Mapping[str, M | TS] | None:
        ...


@jaxtyped
@dataclass
class TrainingHooks:
    """
    Attributes
    ----------

    on_epoch_begin:
    on_epoch_end:
        Typically, should be used to write validation metrics.
    on_step_begin:
    on_step_end:
        Typically, should be used to write training metrics.
    on_training_begin:
        Can be used to reload training training_state from orbax checkpoint. For multi-device environments must return NOT replicated training_state.
    on_training_end:
        Can be used to save models weights, or to notify about training run completion.
    on_error:
        Can be used to process specific error types.


    """

    on_epoch_begin: List[Callable[[int], None]] = dataclasses.field(default_factory=list)
    on_epoch_end: List[OnEpochEnd] = dataclasses.field(default_factory=list)
    on_step_begin: List[Callable[[int], None]] = dataclasses.field(default_factory=list)
    on_step_end: List[OnStepEnd] = dataclasses.field(default_factory=list)
    on_training_begin: List[Callable[[TS], Optional[TS]]] = dataclasses.field(default_factory=list)
    on_training_end: List[Callable[[TS], None]] = dataclasses.field(default_factory=list)
    on_error: List[Callable[[TS, T, Int[Array, "batch classes"], StepType, Exception], None]] = dataclasses.field(default_factory=list)

    def call_on_epoch_begin(self, epoch: int):
        for hook in self.on_epoch_begin:
            hook(epoch)

    def call_on_epoch_end(self, epoch: int, *, validation_metrics: M, training_state: TS) -> Tuple[M, TS]:
        for hook in self.on_epoch_end:
            logs = hook(epoch, validation_metrics=validation_metrics, training_state=training_state)
            if isinstance(logs, Mapping):
                if "training_state" in logs:
                    training_state = training_state
                if "validation_metrics" in logs:
                    validation_metrics = validation_metrics

        return validation_metrics, training_state

    def call_on_step_begin(self, step: int):
        for hook in self.on_step_begin:
            hook(step)

    def call_on_step_end(self, step: int, *, training_metrics: M, training_state: TS) -> Tuple[M, TS]:
        for hook in self.on_step_end:
            logs = hook(step, training_metrics=training_metrics, training_state=training_state)
            if isinstance(hook, Mapping):
                if "training_state" in logs:
                    training_state = training_state
                if "training_metrics" in logs:
                    training_metrics = training_metrics

        return training_metrics, training_state

    def call_on_training_begin(self, training_state: TS) -> TS | None:
        reloaded_state = None
        for hook in self.on_training_begin:
            logs = hook(training_state)
            if isinstance(logs, train_state.TrainState):
                if reloaded_state is not None:
                    raise RuntimeError("Only one reloaded training_state is allowed.")
                reloaded_state = logs

        return reloaded_state

    def call_on_training_end(self, training_state: TS):
        for hook in self.on_training_end:
            hook(training_state)

    @jaxtyped
    @contextmanager
    def catch_error(
        self,
        state: TrainingHooks,
        x_batch: T,
        y_batch: Array,
        step_type: StepType,
    ) -> ContextManager:
        try:
            yield
        except Exception as exception:
            for hook in self.on_error:
                hook(state, x_batch, y_batch, step_type, exception)


@log_exception(ignore_argnames="params")
def save_as_msgpack(params: frozen_dict.FrozenDict, save_path: str = "model.msgpack") -> None:
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
    logging.debug(f"Saving model to {save_path}")
    msgpack_bytes: bytes = frozen_dict.serialization.to_bytes(params)

    with tf.io.gfile.GFile(save_path, "wb+") as file:
        file.write(msgpack_bytes)


@log_exception(ignore_argnames="params")
def load_from_msgpack(params: frozen_dict.FrozenDict, save_path: str = "model.msgpack") -> frozen_dict.FrozenDict:
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
    logging.debug(f"Loading model from {save_path}")

    with tf.io.gfile.GFile(save_path, "rb") as file:
        bytes_data = file.read()

    params = frozen_dict.serialization.from_bytes(params, bytes_data)

    return params


@jaxtyped
def fit_single_device(
    *,
    training_state: TS,
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
        The initial training_state of the training process.
    training_dataset_factory : DatasetFactory
        A factory function that returns the training dataset.
    validation_dataset_factory : DatasetFactory
        A factory function that returns the validation dataset.
    metrics_container_type : Type[M]
        The type of container to store the metrics.
    training_step_func : Callable[[TS, T, Int[Array, "batch classes"]], Tuple[TS, M]]
        A function that performs a single training step. It takes the training training_state, input data, and target data as inputs,
        and returns the updated training training_state and metrics.
    validation_step_func : Callable[[TS, T, Int[Array, "batch classes"]], M]
        A function that performs a single validation step. It takes the training training_state, input data, and target data as inputs,
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
        A tuple containing the training and validation metrics, and the final training training_state parameters.
    """

    if epochs <= 0:
        raise InvalidEpochsNumberError(epochs)

    if hooks is None:
        hooks = TrainingHooks()

    current_step = None
    loaded_state = hooks.call_on_training_begin(training_state)
    if isinstance(loaded_state, train_state.TrainState):
        logging.info("Loaded saved training training_state.")
        training_state = loaded_state
        current_step = 0

    should_stop = False

    training_metrics = metrics_container_type.empty()
    validation_metrics = metrics_container_type.empty()

    for epoch in range(epochs):
        hooks.call_on_epoch_begin(epoch)

        training_dataset = training_dataset_factory()

        if prefetch_buffer_size != 0:
            training_dataset = prefetch_to_device(training_dataset, prefetch_buffer_size)

        if verbose:
            training_dataset = tqdm(
                training_dataset,
                total=num_training_steps,
                desc=f"Epoch {epoch + 1}/{epochs}",
            )
        training_metrics = metrics_container_type.empty()

        for x_batch, y_batch in training_dataset:
            if current_step is not None and current_step < int(training_state.step):
                # Fast-forward reloaded steps
                current_step += 1
                continue

            hooks.call_on_step_begin(int(training_state.step))

            with hooks.catch_error(training_state, x_batch, y_batch, "training"):
                training_state, training_step_metrics_i = training_step_func(training_state, x_batch, y_batch)
            training_metrics = training_metrics.merge(training_step_metrics_i)

            training_metrics, training_state = hooks.call_on_step_end(
                int(training_state.step), training_metrics=training_metrics, training_state=training_state
            )
            should_stop = should_stop_early(training_state)
            if should_stop:
                logging.info("Stopping early")
                break

        if current_step is not None and current_step < int(training_state.step):
            continue

        if verbose:
            logging.info({f"train_{k}": f"{float(v):.3f}"} for k, v in training_metrics.compute().items())

        if should_stop:
            break

        validation_dataset = validation_dataset_factory()
        if prefetch_buffer_size != 0:
            validation_dataset = prefetch_to_device(validation_dataset, prefetch_buffer_size)
        validation_metrics = metrics_container_type.empty()

        for x_batch, y_batch in validation_dataset:
            with hooks.catch_error(training_state, x_batch, y_batch, "validation"):
                validation_step_metrics_i = validation_step_func(training_state, x_batch, y_batch)
            validation_metrics = validation_metrics.merge(validation_step_metrics_i)

        if verbose:
            logging.info({f"val_{k}": f"{float(v):.3f}"} for k, v in validation_metrics.compute().items())

        validation_metrics, training_state = hooks.call_on_epoch_end(
            int(training_state.step), training_state=training_state, validation_metrics=validation_metrics
        )

    params = training_state.params
    training_metrics = training_metrics.compute()
    validation_metrics = validation_metrics.compute()

    hooks.call_on_training_end(training_state)

    return (training_metrics, validation_metrics), params


@jaxtyped
def fit_multi_device(
    *,
    training_state: TS,
    metrics_container_type: Type[M],
    training_step_func: TrainingStep,
    training_dataset_factory: DatasetFactory,
    validation_dataset_factory: DatasetFactory,
    validation_step_func: ValidationStep,
    hooks: TrainingHooks | None = None,
    epochs: int = 1,
    prefetch_buffer_size: int = 2,
    verbose: bool = True,
    num_training_steps: int | None = None,
    skip_shard: bool = False,
) -> MetricsAndParams:
    """
    Parameters
    ----------
    training_state : TS
        The initial training_state of the training process.
    training_dataset_factory : DatasetFactory
        A factory function that returns the training dataset.
    validation_dataset_factory : DatasetFactory
        A factory function that returns the validation dataset.
    metrics_container_type : Type[M]
        The type of container to store the metrics.
    training_step_func : Callable[[TS, T, Int[Array, "batch classes"]], Tuple[TS, M]]
        A function that performs a single training step. It takes the training training_state, input data, and target data as inputs,
        and returns the updated training training_state and metrics.
    validation_step_func : Callable[[TS, T, Int[Array, "batch classes"]], M]
        A function that performs a single validation step. It takes the training training_state, input data, and target data as inputs,
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

    Returns
    -------
    Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]
        A tuple containing the training and validation metrics, and the final training training_state parameters.
    """

    if epochs <= 0:
        raise InvalidEpochsNumberError(epochs)

    if hooks is None:
        hooks = TrainingHooks()

    def shard_x_y(ds: Iterable[Tuple]):
        if skip_shard:
            return ds
        for x, y in ds:
            x = common_utils.shard(x)
            y = common_utils.shard(y)
            yield x, y

    # maybe restore training training_state
    current_step = None
    loaded_state = hooks.call_on_training_begin(training_state)
    if isinstance(loaded_state, train_state.TrainState):
        logging.info("Loaded saved training training_state.")
        training_state = loaded_state
        current_step = 0

    training_state = replicate_state(training_state)

    should_stop = False
    training_metrics = jax_utils.replicate(metrics_container_type.empty())
    validation_metrics = jax_utils.replicate(metrics_container_type.empty())

    for epoch in range(epochs):
        hooks.call_on_epoch_begin(epoch)

        training_dataset = shard_x_y(training_dataset_factory())
        if prefetch_buffer_size != 0:
            training_dataset = jax_utils.prefetch_to_device(training_dataset, prefetch_buffer_size)

        if verbose:
            training_dataset = tqdm(
                training_dataset,
                total=num_training_steps,
                desc=f"Epoch {epoch + 1}/{epochs}...",
            )

        training_metrics = jax_utils.replicate(metrics_container_type.empty())

        for x_batch, y_batch in training_dataset:
            if current_step is not None and current_step < int(current_step):
                # Fast-forward reloaded steps
                current_step += 1
                continue

            hooks.call_on_step_begin(step_number(training_state))

            with hooks.catch_error(jax_utils.unreplicate(training_state), x_batch, y_batch, "training"):
                training_state, training_step_metrics = training_step_func(training_state, x_batch, y_batch)

            training_metrics = training_metrics.merge(training_step_metrics)
            training_metrics, training_state = hooks.call_on_step_end(
                step_number(training_state),
                training_metrics=training_metrics.unreplicate(),
                training_state=jax_utils.unreplicate(training_state),
            )
            should_stop = should_stop_early(jax_utils.unreplicate(training_state))
            if should_stop:
                logging.info("Stopping early")
                break

        if current_step is not None and current_step < int(current_step):
            # Fast-forward reloaded steps
            continue

        if verbose:
            logging.info({f"train_{k}": f"{float(v):.3f}"} for k, v in training_metrics.unreplicate().compute().items())

        if should_stop:
            break

        validation_dataset = shard_x_y(validation_dataset_factory())
        if prefetch_buffer_size != 0:
            validation_dataset = jax_utils.prefetch_to_device(validation_dataset, prefetch_buffer_size)

        validation_metrics = jax_utils.replicate(metrics_container_type.empty())

        for x_batch, y_batch in validation_dataset:
            with hooks.catch_error(jax_utils.unreplicate(training_state), x_batch, y_batch, "validation"):
                validation_step_metrics_i = validation_step_func(training_state, x_batch, y_batch)
            validation_metrics = validation_metrics.merge(validation_step_metrics_i)

        if verbose:
            logging.info({f"val_{k}": f"{float(v):.3f}"} for k, v in validation_metrics.unreplicate().compute().items())

        validation_metrics, training_state = hooks.call_on_epoch_end(
            epoch,
            training_state=jax_utils.unreplicate(training_state),
            validation_metrics=validation_metrics.unreplicate(),
        )

    hooks.call_on_training_end(jax_utils.unreplicate(training_state))
    params = jax_utils.unreplicate(training_state).params
    training_metrics = training_metrics.unreplicate().compute()
    validation_metrics = validation_metrics.unreplicate().compute()

    return (training_metrics, validation_metrics), params


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    write_metrics_frequency: int | None = None,
    tensorboard_logdir: str | None = "tensorboard",
    hyperparams_factory: Callable[[], Dict[str, Any]] | None = None,
    report_progress_frequency: int | None = None,
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
    write_metrics_frequency:
        Number of times per epoch to write metrics/report progress.
    hyperparams_factory:
        If not None, will write return value as hyperparams at beginning of each epoch to
        use with Tensorboard visualization.
    tensorboard_logdir:
        Directory where to write metrics into. If set to None, will not write any tensorboard logs.
    report_progress_frequency:
        Number of times per epoc to report progress to stdout, if set to None no progress report will be generated.

    Returns
    -------

    """

    hooks = TrainingHooks()

    training_writer = clu.metric_writers.create_default_writer(logdir=tensorboard_logdir, collection="training")
    validation_writer = clu.metric_writers.create_default_writer(logdir=tensorboard_logdir, collection="validation")

    def flush(*args, **kwargs):
        training_writer.flush()
        validation_writer.flush()

    hooks.on_training_end.append(flush)

    if report_progress_frequency is not None:
        report_progress = clu.periodic_actions.ReportProgress(
            every_steps=num_training_steps // report_progress_frequency,
            num_train_steps=num_training_steps * epochs,
            writer=training_writer,
            every_secs=None,
        )

        def report_progress_func(step: int, *args, **kwargs):
            report_progress(step)

        hooks.on_step_end.append(report_progress_func)

    def write_training_metrics_fn(step: int, *args, training_metrics: M, **kwargs):
        training_writer.write_scalars(step, training_metrics.compute())

    def write_validation_metrics_fn(epoch: int, *, validation_metrics: M, **kwargs):
        step_num = epoch * num_training_steps
        validation_writer.write_scalars(step_num, validation_metrics.compute())

    if write_metrics_frequency is not None:
        hooks.on_step_end.append(
            clu.periodic_actions.PeriodicCallback(
                on_steps=[1, num_training_steps * epochs],
                every_steps=num_training_steps // write_metrics_frequency,
                callback_fn=write_training_metrics_fn,
                execute_async=True,
            ),
        )
        hooks.on_epoch_end.append(
            UncheckedPeriodicCallback(
                on_steps=[num_training_steps * epochs],
                every_steps=num_training_steps // write_metrics_frequency,
                callback_fn=write_validation_metrics_fn,
                execute_async=True,
            ),
        )

    if hyperparams_factory is not None:

        def write_hparams(*args, **kwargs):
            training_writer.write_hparams(hyperparams_factory())

        hooks.on_training_begin.append(write_hparams)

    return hooks


def replicate_state(state: TS) -> TS:
    state = jax_utils.replicate(state)
    if hasattr(state, "dropout_key"):
        state.replace(dropout_key=common_utils.shard_prng_key(jax_utils.unreplicate(state.dropout_key)))
    return state


def step_number(state: TS):
    return int(jax_utils.unreplicate(state.step))


def should_stop_early(state: TS) -> bool:
    return (
        hasattr(state, "early_stopping")
        and isinstance(state.early_stopping, EarlyStopping)
        and state.early_stopping.should_stop
    )
