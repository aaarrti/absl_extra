from __future__ import annotations

import functools
import json
from functools import wraps
from importlib import util
from typing import (
    Callable,
    List,
    Mapping,
    NamedTuple,
    TypeVar,
    Protocol,
    ParamSpecArgs,
    ParamSpecKwargs,
    TYPE_CHECKING,
    Dict,
)

from absl import app, flags, logging

from absl_extra.notifier import BaseNotifier, LoggingNotifier

T = TypeVar("T", bound=Callable)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "task", default=None, required=True, help="Name of the function to execute."
)

if util.find_spec("pymongo"):
    from pymongo import MongoClient
    from pymongo.collection import Collection
else:
    Collection = None
    logging.warning("pymongo not installed.")

if util.find_spec("ml_collections"):
    from ml_collections import ConfigDict
    from ml_collections import config_flags
else:
    logging.warning("ml_collections not installed")
    ConfigDict = None


if TYPE_CHECKING:
    from absl_extra.src.callbacks import CallbackFn


class MongoConfig(NamedTuple):
    uri: str
    db_name: str
    collection: str


class _ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, name: str, notifier: BaseNotifier):
        self.name = name
        self.notifier = notifier

    def handle(self, exception: Exception) -> None:
        self.notifier.notify_task_failed(self.name, exception)


class TaskFn(Protocol):
    def __call__(self, config: ConfigDict = None, db: Collection = None) -> None:
        ...


TASK_STORE: Dict[str, Callable[[], None]] = dict()


class NonExistentTaskError(RuntimeError):
    def __init__(self, task: str):
        super().__init__(
            f"Unknown task {task}, registered are {list(TASK_STORE.keys())}"
        )


def make_task_func(
    func: Callable[[], None],
    *,
    name: str,
    notifier: BaseNotifier | Callable[[], BaseNotifier],
    config_file: str | None,
    init_callbacks: List[CallbackFn],
    post_callbacks: List[CallbackFn],
    db: Collection | None,
) -> None:
    if isinstance(name, Callable):
        _name = name()
    else:
        _name = name

    @functools.wraps(func)
    def wrapper():
        kwargs = {}

        if util.find_spec("ml_collections") and config_file is not None:
            config = config_flags.DEFINE_config_file("config", default=config_file)
            config = config.value
            kwargs["config"] = config
        else:
            config = None
        if db is not None:
            kwargs["db"] = db

        for hook in init_callbacks:
            hook(_name, notifier=notifier, config=config, db=db)

        func()

        for hook in post_callbacks:
            hook(_name, notifier=notifier, config=config, db=db)

    return wrapper


def register_task(
    name: str,
    notifier: BaseNotifier | Callable[[], BaseNotifier] | None = None,
    config_file: str | None = None,
    mongo_config: MongoConfig | Mapping[str, ...] | None = None,
    init_callbacks: List[CallbackFn] = None,
    post_callbacks: List[CallbackFn] = None,
) -> Callable[[TaskFn], None]:
    """

    Parameters
    ----------
    fn:
        Function to execute.
    name:
        Name to be used for lifecycle reporting.
    init_callbacks:
        List of callback, which must be called on initialization.
        By default, will print parsed absl.flags and ml_collection.ConfigDict to stdout.

    Returns
    -------

    """
    from absl_extra.callbacks import (
        log_params_callback,
        log_startup_callback,
        log_shutdown_callback,
    )

    if isinstance(notifier, Callable):
        notifier = notifier()
    if notifier is None:
        notifier = LoggingNotifier()

    if util.find_spec("pymongo") and mongo_config is not None:
        if isinstance(mongo_config, Mapping):
            mongo_config = MongoConfig(**mongo_config)
        db = (
            MongoClient(mongo_config.uri)
            .get_database(mongo_config.db_name)
            .get_collection(mongo_config.collection)
        )
    else:
        db = None

    if init_callbacks is None:
        init_callbacks = [log_params_callback, log_startup_callback]

    if post_callbacks is None:
        post_callbacks = [log_shutdown_callback]

    app.install_exception_handler(_ExceptionHandlerImpl(name, notifier))

    def decorator(func: TaskFn) -> None:
        TASK_STORE[name] = functools.partial(
            make_task_func,
            name=name,
            notifier=notifier,
            init_callbacks=init_callbacks,
            post_callbacks=post_callbacks,
            db=db,
            config_file=config_file,
        )(func)

    return decorator


def select_main(_):
    task_name = FLAGS.task
    if task_name not in TASK_STORE:
        raise NonExistentTaskError(task_name)
    func = TASK_STORE[task_name]
    func()


def run():
    app.run(select_main)
