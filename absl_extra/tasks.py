from __future__ import annotations

import json
from functools import wraps, partial
from importlib import util
from typing import Callable, NamedTuple, TypeVar, Mapping

from absl import app, flags, logging

from absl_extra.notifier import BaseNotifier

T = TypeVar("T", bound=Callable, covariant=True)
FLAGS = flags.FLAGS

flags.DEFINE_string("task", default="main", help="Name of task to run.")

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


_TASKS = {}


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
    if isinstance(name, Callable):
        name = name()
    if name in _TASKS:
        raise RuntimeError(f"Can't register 2 tasks with same name: {name}")
    _TASKS[name] = fn


def hook_task(
    task: Callable[[...], None],
    app_name: str,
    task_name: str,
    notifier: BaseNotifier,
    config_file: str | None,
) -> Callable[[...], None]:
    @wraps(task)
    def wrapper(*, db=None):
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
        if util.find_spec("tensorflow") is not None:
            import tensorflow as tf

            logging.info(f"TF decices -> {tf.config.list_logical_devices()}")

        notifier.notify_job_started(f"{app_name}.{task_name}")

        kwargs = {}
        if db is not None:
            kwargs["db"] = db
        if config is not None:
            kwargs["config"] = config

        ret_val = task(**kwargs)
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

    def pseudo_main(cmd, **kwargs) -> None:
        TASKS[FLAGS.task](**kwargs)

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
    global _TASKS
    TASKS = {
        k: hook_task(
            task=v,
            task_name=k,
            notifier=notifier,
            config_file=config_file,
            app_name=app_name,
        )
        for k, v in _TASKS.items()
    }
    app.run(partial(pseudo_main, db=db))
