from __future__ import annotations

import json
from functools import wraps, partial
from importlib import util
from typing import Callable, NamedTuple, TypeVar, Mapping, List

from absl import app, flags, logging

from absl_extra.notifier import BaseNotifier

T = TypeVar("T", bound=Callable, covariant=True)
FLAGS = flags.FLAGS

if util.find_spec("pymongo"):
    from pymongo import MongoClient
else:
    logging.warning("pymongo not installed.")

if util.find_spec("ml_collections"):
    from ml_collections import config_flags
else:
    logging.warning("ml_collections not installed")


class MongoConfig(NamedTuple):
    uri: str
    db_name: str
    collection: str | None = None


class _ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, name: str, notifier: BaseNotifier):
        self.name = name
        self.notifier = notifier

    def handle(self, exception: Exception) -> None:
        self.notifier.notify_job_failed(self.name, exception)


class TaskT(NamedTuple):
    fn: Callable
    name: str | Callable[[], str]
    init_callbacks: List[Callable[[...], None]] = []


_TASK_STORE = None


def register_task(
    fn: Callable[[str], None], name: str | Callable[[], str] = "main"
) -> None:
    """

    Parameters
    ----------
    fn
    name

    Returns
    -------

    """
    global _TASK_STORE
    _TASK_STORE = TaskT(fn=fn, name=name)


def pseudo_main(
    task: TaskT,
    app_name: str,
    notifier: BaseNotifier,
    config_file: str | None,
) -> Callable[[...], None]:
    @wraps(task)
    def wrapper(*, db=None):
        if isinstance(task.name, Callable):
            task_name = task.name()
        else:
            task_name = task.name

        logging.info("-" * 50)
        logging.info(
            f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
        )
        if util.find_spec("ml_collections") and config_file is not None:
            config = config_flags.DEFINE_config_file("config", default=config_file)
        else:
            config = None

        if config is not None:
            config = config.value
            logging.info(
                f"Config: {json.dumps(dict(config), sort_keys=True, indent=4)}"
            )
        logging.info("-" * 50)
        notifier.notify_job_started(f"{app_name}.{task_name}")

        kwargs = {}
        if db is not None:
            kwargs["db"] = db
        if config is not None:
            kwargs["config"] = config

        ret_val = task.fn(**kwargs)
        notifier.notify_job_finished(f"{app_name}.{task_name}")
        return ret_val

    return wrapper


def run(
    app_name: str | Callable[[], str] = "app",
    notifier: BaseNotifier | None = None,
    config_file: str | None = None,
    mongo_config: MongoConfig | Mapping[str, ...] | None = None,
):
    """

    Parameters
    ----------
    app_name
    notifier
    config_file
    mongo_config

    Returns
    -------

    """
    if notifier is None:
        notifier = BaseNotifier()

    if util.find_spec("pymongo") and mongo_config is not None:
        if isinstance(mongo_config, Mapping):
            mongo_config = MongoConfig(**mongo_config)
        db = MongoClient(mongo_config.uri).get_database(mongo_config.db_name)
        if mongo_config.collection is not None:
            db = db.get_collection(mongo_config.collection)
    else:
        db = None

    app.install_exception_handler(_ExceptionHandlerImpl(app_name, notifier))

    task_fn = pseudo_main(
        task=_TASK_STORE,
        notifier=notifier,
        config_file=config_file,
        app_name=app_name,
    )
    app.run(partial(task_fn, db=db))
