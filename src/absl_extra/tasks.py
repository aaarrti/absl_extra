from __future__ import annotations

import functools
from importlib import util
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Protocol,
    TypeVar,
)

from absl import app, flags, logging

from absl_extra.notifier import BaseNotifier, LoggingNotifier
from absl_extra.dataclass import dataclass

T = TypeVar("T", bound=Callable)
FLAGS = flags.FLAGS
flags.DEFINE_string("task", default="main", help="Name of the function to execute.")
flags.DEFINE_enum(
    "log_level",
    enum_values=["INFO", "DEBUG", "ERROR", "WARNING"],
    default="INFO",
    help="Logging level to use. If None, no auto-setup will be executed.",
)

if util.find_spec("pymongo"):
    from pymongo import MongoClient
    from pymongo.collection import Collection
else:
    Collection = type(None)
    logging.warning("pymongo not installed.")

if TYPE_CHECKING:
    from absl_extra.callbacks import CallbackFn


@dataclass
class MongoConfig:
    uri: str
    db_name: str
    collection: str


class _ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, name: str, notifier: BaseNotifier):
        self.name = name
        self.notifier = notifier

    def handle(self, exception: Exception) -> None:
        self.notifier.notify_task_failed(self.name, exception)


class _TaskFn(Protocol):
    def __call__(self, *, db: Collection = None, **kwargs) -> None:
        ...


_TASK_STORE: Dict[str, Callable[[...], None]] = dict()  # type: ignore


class NonExistentTaskError(RuntimeError):
    def __init__(self, task: str):
        super().__init__(
            f"Unknown task {task}, registered are {list(_TASK_STORE.keys())}"
        )


def _make_task_func(
    func: _TaskFn,
    *,
    name: str,
    notifier: BaseNotifier | Callable[[], BaseNotifier],
    init_callbacks: List[CallbackFn],
    post_callbacks: List[CallbackFn],
    db_factory=None,
) -> _TaskFn:
    _name = name

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        app.install_exception_handler(_ExceptionHandlerImpl(name, notifier))  # type: ignore
        kwargs = {}
        if db_factory is not None:
            db = db_factory()
            kwargs["db"] = db

        for hook in init_callbacks:
            hook(_name, notifier=notifier, **kwargs)

        func(*args, **kwargs)

        for hook in post_callbacks:
            hook(_name, notifier=notifier, **kwargs)

    return wrapper


def register_task(
    *,
    name: str = "main",
    notifier: BaseNotifier | Callable[[], BaseNotifier] | None = None,
    mongo_config: MongoConfig | Mapping[str, Any] | None = None,
    init_callbacks: List[CallbackFn] | None = None,
    post_callbacks: List[CallbackFn] | None = None,
) -> Callable[[_TaskFn], None]:
    """
    Parameters
    ----------
    name : str, optional
        The name of the task. Default is "main".
    notifier : BaseNotifier | Callable[[], BaseNotifier] | None, optional
        The notifier object or callable that returns a notifier object. Default is None.
    mongo_config : MongoConfig | Mapping[str, Any] | None, optional
        The configuration object for MongoDB or a mapping of configuration values. Default is None.
    init_callbacks : List[CallbackFn] | None, optional
        The list of callback functions to be executed during task initialization. Default is None.
    post_callbacks : List[CallbackFn] | None, optional
        The list of callback functions to be executed after the task completes. Default is None.

    Returns
    -------
    Callable[[_TaskFn], None]
        The decorator function that registers the task.
    """
    from absl_extra.callbacks import DEFAULT_INIT_CALLBACKS, DEFAULT_POST_CALLBACK

    if isinstance(notifier, Callable):  # type: ignore
        notifier = notifier()  # type: ignore
    if notifier is None:
        notifier = LoggingNotifier()

    kwargs = {}

    if util.find_spec("pymongo") and mongo_config is not None:
        if isinstance(mongo_config, Mapping):
            mongo_config = MongoConfig(**mongo_config)
        db_factory = lambda: (  # noqa
            MongoClient(mongo_config.uri)
            .get_database(mongo_config.db_name)
            .get_collection(mongo_config.collection)
        )
        kwargs["db_factory"] = db_factory

    if init_callbacks is None:
        init_callbacks = DEFAULT_INIT_CALLBACKS  # type: ignore

    if post_callbacks is None:
        post_callbacks = DEFAULT_POST_CALLBACK  # type: ignore

    def decorator(func: _TaskFn) -> None:
        _TASK_STORE[name] = functools.partial(  # type: ignore
            _make_task_func,
            name=name,
            notifier=notifier,
            init_callbacks=init_callbacks,
            post_callbacks=post_callbacks,
            **kwargs,
        )(func)

    return decorator


def run(argv: List[str] | None = None, **kwargs):
    """
    Parameters
    ----------
    argv:
        CLI args passed to absl.app.run
    kwargs:
        Kwargs passed to entrypoint function.

    Returns
    -------

    """

    def select_main(_):
        task_name = FLAGS.task
        if task_name not in _TASK_STORE:
            raise NonExistentTaskError(task_name)
        func = _TASK_STORE[task_name]
        func(**kwargs)

    app.run(select_main, argv=argv)
