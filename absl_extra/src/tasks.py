from __future__ import annotations

import json
from functools import wraps
from importlib import util
from typing import Callable, NamedTuple, TypeVar, Mapping, List

from absl import app, flags, logging

from absl_extra.src.notifier import BaseNotifier

T = TypeVar("T", bound=Callable, covariant=True)
FLAGS = flags.FLAGS

if util.find_spec("pymongo"):
    from pymongo import MongoClient
    from pymongo.collection import Collection
else:
    Collection = None
    logging.warning("pymongo not installed.")

if util.find_spec("ml_collections"):
    from ml_collections import config_flags
else:
    logging.warning("ml_collections not installed")


class MongoConfig(NamedTuple):
    uri: str
    db_name: str
    collection: str


class _ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, name: str, notifier: BaseNotifier):
        self.name = name
        self.notifier = notifier

    def handle(self, exception: Exception) -> None:
        self.notifier.notify_job_failed(self.name, exception)


class TaskT(NamedTuple):
    fn: Callable
    name: str | Callable[[str, ...], str]
    init_callbacks: List[Callable[[...], None]] = []


_TASK_STORE = None


def register_task(
    fn: Callable[[str], None],
    name: str | Callable[[], str] = "main",
    init_callback: List[Callable[[...], None]] = None,
) -> None:
    """

    Parameters
    ----------
    fn:
        Function to execute.
    name:
        Name to be used for lifecycle reporting.
    init_callback:
        List of callback, which must be called on initialization.
        By default, will print parsed absl.flags and ml_collection.ConfigDict to stdout.

    Returns
    -------

    """
    global _TASK_STORE
    _TASK_STORE = TaskT(fn=fn, name=name)


def pseudo_main(
    task: TaskT,
    app_name: str,
    notifier: BaseNotifier,
    config_file: str | None = None,
    db: Collection | None = None,
) -> Callable[[str, ...], None]:
    @wraps(task.fn)
    def wrapper(cmd: str):
        if isinstance(task.name, Callable):
            task_name = task.name()
        else:
            task_name = task.name

        logging.info("-" * 50)
        logging.info(
            f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
        )

        kwargs = {}

        if util.find_spec("ml_collections") and config_file is not None:
            config = config_flags.DEFINE_config_file("config", default=config_file)
            config = config.value
            logging.info(
                f"Config: {json.dumps(config.to_dict(), sort_keys=True, indent=4)}"
            )
            kwargs["config"] = config
        logging.info("-" * 50)
        if db is not None:
            kwargs["db"] = db
        notifier.notify_job_started(f"{app_name}.{task_name}")
        ret_val = task.fn(cmd, **kwargs)
        notifier.notify_job_finished(f"{app_name}.{task_name}")
        return ret_val

    return wrapper


def run(
    app_name: str | Callable[[], str] = "app",
    notifier: BaseNotifier | Callable[[], BaseNotifier] | None = None,
    config_file: str | None = None,
    mongo_config: MongoConfig | Mapping[str, ...] | None = None,
) -> None:
    """

    Parameters
    ----------
    app_name:
        Name of the task.
    notifier:
        Notifier instance, which monitors execution state.
    config_file:
        Optional, path to ml_collection config file.
    mongo_config:
        Optional, NamedTuple or Dict with MongoDB connection credentials.

    Returns
    -------

    """
    if isinstance(notifier, Callable):
        notifier = notifier()
    if notifier is None:
        notifier = BaseNotifier()

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

    app.install_exception_handler(_ExceptionHandlerImpl(app_name, notifier))

    task_fn = pseudo_main(
        task=_TASK_STORE,
        notifier=notifier,
        config_file=config_file,
        app_name=app_name,
        db=db,
    )
    app.run(task_fn)
