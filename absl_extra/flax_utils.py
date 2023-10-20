from __future__ import annotations

import zlib
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    no_type_check,
    overload,
)

import jax.numpy as jnp
import jax.random
from absl import logging
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import prefetch_to_device, replicate, unreplicate
from flax.serialization import from_bytes, msgpack_restore, to_bytes
from flax.struct import dataclass, field
from flax.training import common_utils, train_state
from flax.training.early_stopping import EarlyStopping
from tqdm.auto import tqdm

from absl_extra.logging_utils import log_exception
from absl_extra.typing_utils import ParamSpec

if TYPE_CHECKING:
    from clu.metrics import Collection

    # This one should not be directly subclassed
    class TrainStateContainer(train_state.TrainState):
        dropout_key: jax.random.KeyArray | None
        early_stopping: EarlyStopping | None

    P = ParamSpec("P")
    S = TypeVar("S", bound=Sequence)
    C = TypeVar("C", bound=Callable)
    DatasetFactory = Callable[[], Iterable[Tuple[jnp.ndarray, jnp.ndarray]]]

    TS = TypeVar("TS", bound=TrainStateContainer)

    M = TypeVar("M", bound=Collection, contravariant=True)
    ValidationStep = Callable[[TS, jnp.ndarray, jnp.ndarray], M]
    TrainingStep = Callable[[TS, jnp.ndarray, jnp.ndarray], Tuple[TS, M]]
    MetricsAndParams = Tuple[Tuple[Dict[str, float], Dict[str, float]], FrozenDict]
    StepType = Literal["training", "validation"]
    CustomReplication = Tuple[Callable[[TS], TS], Callable[[TS], TS]]

    class OnStepEnd(Protocol):
        def __call__(self, step: int, *, training_metrics: M, training_state: TS) -> Mapping[str, M | TS] | None:
            ...

    class OnEpochEnd(Protocol):
        def __call__(self, epoch: int, *, validation_metrics: M, training_state: TS) -> Mapping[str, M | TS] | None:
            ...

    class OnError(Protocol):
        def __call__(
            self,
            *,
            training_state: TS,
            x_batch: jnp.ndarray,
            y_batch: jnp.ndarray,
            step_type: StepType,
            exception: Exception,
        ) -> bool | None:
            ...


class ParamReplication(NamedTuple):
    replicate: Callable[[TS], TS]
    un_replicate: Callable[[TS], TS]


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
        Can be used to reload training training_state from orbax checkpoint.
        For multi-device environments must return NOT replicated training_state.
    on_training_end:
        Can be used to save models weights, or to notify about training run completion.
    on_error:
        Can be used to process specific error types.



    Examples
    --------
    >>> import clu.metric_writers
    >>> num_train_steps=1000
    >>> epochs = 5
    >>> hooks = TrainingHooks()
    >>> training_writer = clu.metric_writers.create_default_writer(logdir="tensorboard", collection="training")
    >>> validation_writer = clu.metric_writers.create_default_writer(logdir="tensorboard", collection="validation")
    >>> def flush(*args, **kwargs):
    ...     training_writer.flush()
    ...     validation_writer.flush()
    >>> hooks.on_training_end.append(flush)
    >>> report_progress = clu.periodic_actions.ReportProgress(every_steps=100, num_train_steps=num_train_steps * epochs,
     ... writer=training_writer, every_secs=None)
    >>>  def report_progress_func(step: int, *args, **kwargs):
    ...      report_progress(step)
    >>> hooks.on_step_end.append(report_progress_func)
    >>> def write_training_metrics_fn(step: int, *args, training_metrics, **kwargs):
    ...     training_writer.write_scalars(step, training_metrics.compute())
    >>> def write_validation_metrics_fn(epoch: int, *, validation_metrics, **kwargs):
    ...     step_num = epoch * num_train_steps
    ...     validation_writer.write_scalars(step_num, validation_metrics.compute())
    >>> hooks.on_step_end.append(
    ...     clu.periodic_actions.PeriodicCallback(
    ...         on_steps=[1, num_train_steps * epochs],
    ...         every_steps=100,
    ...         callback_fn=write_training_metrics_fn,
    ...         execute_async=True,
    ...     ),
    ... )
    >>> hooks.on_epoch_end.append(write_validation_metrics_fn)
    >>> def write_hparams(*args, **kwargs):
    ...     training_writer.write_hparams({"learning_rate": 1e-3, "ema": 0.99})
    >>> hooks.on_training_begin.append(write_hparams)
    >>> fit_single_device(hooks=hooks, ...)
    """

    on_epoch_begin: List[Callable[[int], None]] = field(pytree_node=False, default_factory=list)
    on_epoch_end: List[OnEpochEnd] = field(pytree_node=False, default_factory=list)
    on_step_begin: List[Callable[[int], None]] = field(pytree_node=False, default_factory=list)
    on_step_end: List[OnStepEnd] = field(pytree_node=False, default_factory=list)
    on_training_begin: List[Callable[[TS], Optional[TS]]] = field(pytree_node=False, default_factory=list)
    on_training_end: List[Callable[[TS], None]] = field(pytree_node=False, default_factory=list)
    on_error: List[OnError] = field(pytree_node=False, default_factory=list)

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
            if logs is not None and isinstance(hook, Mapping):
                if "training_state" in logs:
                    training_state = training_state
                if "training_metrics" in logs:
                    training_metrics = training_metrics

        return training_metrics, training_state

    def call_on_training_begin(self, training_state: TS):
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

    @contextmanager
    def catch_error(
        self,
        training_state: TS,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        step_type: StepType,
    ):
        try:
            yield
        except Exception as exception:
            handled = False
            for hook in self.on_error:
                retval = hook(
                    training_state=training_state,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    step_type=step_type,
                    exception=exception,
                )
                if isinstance(retval, bool) and retval:
                    handled = handled or retval
            if not handled:
                raise


def combine_hooks(*hooks: TrainingHooks) -> TrainingHooks:
    combined_hooks = TrainingHooks()

    for h in hooks:
        combined_hooks.on_training_begin.extend(h.on_training_begin)
        combined_hooks.on_training_end.extend(h.on_training_end)
        combined_hooks.on_epoch_begin.extend(h.on_epoch_begin)
        combined_hooks.on_epoch_end.extend(h.on_epoch_end)
        combined_hooks.on_step_begin.extend(h.on_step_begin)
        combined_hooks.on_step_end.extend(h.on_step_end)
        combined_hooks.on_error.extend(h.on_error)

    return combined_hooks


@log_exception(ignore_argnames="params")
def save_as_msgpack(
    params: FrozenDict, save_path: str = "model.msgpack", compression: Literal["GZIP"] | None = None
) -> None:
    """
    Parameters
    ----------
    params : frozen_dict.FrozenDict
        The frozen dictionary object that contains the parameters to be saved.
    save_path : str, optional
        The file path where the msgpack file will be saved. Default is "model.msgpack".
    compression:
        If set to GZIP, will compress bytes using gzip before saving to file-system.

    Returns
    -------
    None
        This method does not return any value.
    """
    logging.debug(f"Saving to {save_path}")
    msgpack_bytes: bytes = to_bytes(params)

    if compression == "GZIP":
        msgpack_bytes = zlib.compress(msgpack_bytes)

    try:
        import tensorflow as tf

        with tf.io.gfile.GFile(save_path, "wb+") as file:
            file.write(msgpack_bytes)
    except (ModuleNotFoundError, ImportError):
        logging.error("Failed to import tensorflow.io, falling back to local file-system")
        with open(save_path, "wb+") as file:
            file.write(msgpack_bytes)


@overload
def load_from_msgpack(params: None, save_path: str, compression: Literal["GZIP"] | None = None) -> Dict[str, Any]:
    ...


@overload
def load_from_msgpack(params: FrozenDict, save_path: str, compression: Literal["GZIP"] | None = None) -> FrozenDict:
    ...


@log_exception(ignore_argnames="params")
def load_from_msgpack(
    params: FrozenDict | None, save_path: str = "model.msgpack", compression: Literal["GZIP"] | None = None
) -> FrozenDict | Dict[str, Any]:
    """
    Load model parameters from a msgpack file.

    Parameters
    ----------
    params : frozen_dict.FrozenDict
        The original parameters of the model.
    save_path : str, optional
        The path to the msgpack file containing the serialized parameters.
        Default is "model.msgpack".
    compression:
        Set to GZIP if bytes were GZIP compressed before saving to msgpack.

    Returns
    -------
    params : frozen_dict.FrozenDict
        The loaded parameters.

    """
    logging.debug(f"Loading model from {save_path}")

    try:
        import tensorflow as tf

        with tf.io.gfile.GFile(save_path, "rb") as file:
            bytes_data = file.read()

    except (ModuleNotFoundError, ImportError):
        logging.error("Failed to import tensorflow.io, falling back to local file-system")
        with open(save_path, "rb") as file:
            bytes_data = file.read()

    if compression == "GZIP":
        bytes_data = zlib.decompress(bytes_data)

    if params is not None:
        params = from_bytes(params, bytes_data)
    else:
        params = msgpack_restore(bytes_data)

    return params  # type: ignore


def fit(
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
    param_replication: ParamReplication | None = None,
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
        A function that performs a single training step. It takes the training training_state,
        input data, and target data as inputs, and returns the updated training training_state and metrics.
    validation_step_func : Callable[[TS, T, Int[Array, "batch classes"]], M]
        A function that performs a single validation step. It takes the training training_state, input data,
        and target data as inputs, and returns the metrics.
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
    param_replication:

    Returns
    -------
    Tuple[Tuple[Dict[str, float], Dict[str, float]], frozen_dict.FrozenDict]
        A tuple containing the training and validation metrics, and the final training training_state parameters.
    """
    if epochs <= 0:
        raise RuntimeError(f"Epochs must be greater than 0, but found {epochs}")

    if hooks is None:
        hooks = TrainingHooks()

    if jax.device_count() == 1:
        return fit_single_device(
            training_state=training_state,
            metrics_container_type=metrics_container_type,
            training_step_func=training_step_func,
            training_dataset_factory=training_dataset_factory,
            validation_dataset_factory=validation_dataset_factory,
            validation_step_func=validation_step_func,
            epochs=epochs,
            verbose=verbose,
            hooks=hooks,
            num_training_steps=num_training_steps,
        )
    else:
        return fit_multi_device(
            training_state=training_state,
            metrics_container_type=metrics_container_type,
            training_step_func=training_step_func,
            training_dataset_factory=training_dataset_factory,
            validation_dataset_factory=validation_dataset_factory,
            validation_step_func=validation_step_func,
            hooks=hooks,
            epochs=epochs,
            prefetch_buffer_size=prefetch_buffer_size,
            verbose=verbose,
            num_training_steps=num_training_steps,
            param_replication=param_replication,
        )


def fit_single_device(
    *,
    training_state: TS,
    metrics_container_type: Type[M],
    training_step_func: TrainingStep,
    training_dataset_factory: DatasetFactory,
    validation_dataset_factory: DatasetFactory,
    validation_step_func: ValidationStep,
    epochs: int,
    verbose: bool,
    hooks: TrainingHooks,
    num_training_steps: int | None,
) -> MetricsAndParams:
    current_step = None
    loaded_state = hooks.call_on_training_begin(training_state)
    if isinstance(loaded_state, train_state.TrainState):
        logging.info("Loaded saved training training_state.")
        training_state = loaded_state  # type: ignore
        current_step = 0

    should_stop = False

    training_metrics: M = metrics_container_type.empty()
    validation_metrics: M = metrics_container_type.empty()

    for epoch in range(epochs):
        hooks.call_on_epoch_begin(epoch)

        training_dataset = training_dataset_factory()

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
            logging.info(format_metrics(training_metrics, prefix="training"))

        if should_stop:
            break

        validation_dataset = validation_dataset_factory()
        validation_metrics = metrics_container_type.empty()

        for x_batch, y_batch in validation_dataset:
            with hooks.catch_error(training_state, x_batch, y_batch, "validation"):
                validation_step_metrics_i = validation_step_func(training_state, x_batch, y_batch)
            validation_metrics = validation_metrics.merge(validation_step_metrics_i)

        if verbose:
            logging.info(format_metrics(validation_metrics, prefix="validation"))

        validation_metrics, training_state = hooks.call_on_epoch_end(
            epoch, training_state=training_state, validation_metrics=validation_metrics
        )

    params = training_state.params
    training_metrics = training_metrics.compute()
    validation_metrics = validation_metrics.compute()

    hooks.call_on_training_end(training_state)

    return (training_metrics, validation_metrics), params


def fit_multi_device(
    *,
    training_state: TS,
    metrics_container_type: Type[M],
    training_step_func: TrainingStep,
    training_dataset_factory: DatasetFactory,
    validation_dataset_factory: DatasetFactory,
    validation_step_func: ValidationStep,
    hooks: TrainingHooks,
    epochs,
    prefetch_buffer_size,
    verbose: bool = True,
    num_training_steps: int | None,
    param_replication: ParamReplication | None = None,
) -> MetricsAndParams:
    if epochs <= 0:
        raise RuntimeError(f"Epochs must be greater than 0, but found {epochs}")

    if hooks is None:
        hooks = TrainingHooks()

    if param_replication is None:
        param_replication = make_default_param_sharding()

    # maybe restore training training_state
    current_step = None
    loaded_state = hooks.call_on_training_begin(training_state)
    if isinstance(loaded_state, train_state.TrainState):
        logging.info("Loaded saved training training_state.")
        training_state = loaded_state  # type: ignore
        current_step = 0

    should_stop = False
    training_metrics: M = replicate(metrics_container_type.empty())
    validation_metrics: M = replicate(metrics_container_type.empty())

    for epoch in range(epochs):
        hooks.call_on_epoch_begin(epoch)

        training_dataset = shard_x_y(training_dataset_factory())
        if prefetch_buffer_size != 0:
            training_dataset = prefetch_to_device(training_dataset, prefetch_buffer_size)

        if verbose:
            training_dataset = tqdm(
                training_dataset,
                total=num_training_steps,
                desc=f"Epoch {epoch + 1}/{epochs}...",
            )

        training_metrics = metrics_container_type.empty()

        for x_batch, y_batch in training_dataset:
            if current_step is not None and current_step < int(current_step):
                # Fast-forward reloaded steps
                current_step += 1
                continue

            hooks.call_on_step_begin(int(training_state.step))

            replicated_state = param_replication.replicate(training_state)
            with hooks.catch_error(training_state, x_batch, y_batch, "training"):
                replicated_state, training_step_metrics = training_step_func(replicated_state, x_batch, y_batch)

            training_metrics = training_metrics.merge(training_step_metrics.unreplicate())
            training_state = param_replication.un_replicate(replicated_state)

            training_metrics, training_state = hooks.call_on_step_end(
                int(training_state.step),
                training_metrics=training_metrics,
                training_state=training_state,
            )
            should_stop = should_stop_early(training_state)
            if should_stop:
                logging.info("Stopping early")
                break

        if current_step is not None and current_step < int(current_step):
            # Fast-forward reloaded steps
            continue

        if verbose:
            logging.info(format_metrics(training_metrics, prefix="training"))

        if should_stop:
            break

        validation_dataset = shard_x_y(validation_dataset_factory())
        if prefetch_buffer_size != 0:
            validation_dataset = prefetch_to_device(validation_dataset, prefetch_buffer_size)

        validation_metrics = metrics_container_type.empty()
        replicated_state = param_replication.replicate(training_state)

        for x_batch, y_batch in validation_dataset:
            with hooks.catch_error(training_state, x_batch, y_batch, "validation"):
                validation_step_metrics_i: M = validation_step_func(replicated_state, x_batch, y_batch)
            validation_metrics = validation_metrics.merge(validation_step_metrics_i.unreplicate())

        if verbose:
            logging.info(format_metrics(validation_metrics, prefix="validation"))

        training_state = param_replication.un_replicate(replicated_state)
        validation_metrics, training_state = hooks.call_on_epoch_end(
            epoch,
            training_state=training_state,
            validation_metrics=validation_metrics,
        )

    hooks.call_on_training_end(training_state)
    training_metrics = training_metrics.compute()
    validation_metrics = validation_metrics.compute()

    return (training_metrics, validation_metrics), training_state.params


def format_metrics(metrics: M, prefix: str) -> Dict[str, str]:
    return {f"{prefix}_{k}": f"{float(v):.3f}" for k, v in metrics.compute().items()}


# ---------------------- distributed utils ------------------------
def shard_x_y(ds: Iterable[Tuple]):
    for x, y in ds:
        x = common_utils.shard(x)
        y = common_utils.shard(y)
        yield x, y


def should_stop_early(state: TS) -> bool:
    return state.early_stopping is not None and state.early_stopping.should_stop


@no_type_check
def make_default_param_sharding() -> ParamReplication:
    def replicate_fn(ts: TS):
        replicated_state = replicate(ts)
        if hasattr(ts, "dropout_key") and ts.dropout_key is not None:
            replicated_state = replicated_state.replace(dropout_key=common_utils.shard_prng_key(ts.dropout_key))
        return replicated_state

    return ParamReplication(replicate=replicate_fn, un_replicate=unreplicate)
