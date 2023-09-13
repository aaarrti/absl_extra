from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Dict, List

import toolz
from absl import app, flags

from absl_extra.notifier import BaseNotifier, LoggingNotifier
from absl_extra.typing_utils import ParamSpec

P = ParamSpec("P")
FLAGS = flags.FLAGS

if TYPE_CHECKING:
    from absl_extra.callbacks import CallbackFn


class _ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, name: str, notifier: BaseNotifier):
        self.name = name
        self.notifier = notifier

    def handle(self, exception: Exception) -> None:
        self.notifier.notify_task_failed(self.name, exception)


_TASK_STORE: Dict[str, Callable[[...], None]] = dict()  # type: ignore


@toolz.curry
def make_task_func(
    func: Callable[P, None],
    *,
    name: str,
    notifier: BaseNotifier,
    init_callbacks: List[CallbackFn],
    post_callbacks: List[CallbackFn],
) -> Callable[P, None]:
    _name = name

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        app.install_exception_handler(_ExceptionHandlerImpl(name, notifier))  # type: ignore
        for hook in init_callbacks:
            hook(_name, notifier=notifier, **kwargs)

        func(*args, **kwargs)

        for hook in post_callbacks:
            hook(_name, notifier=notifier, **kwargs)

    return wrapper


@toolz.curry
def register_task(
    func: Callable[P, None],
    *,
    name: str = "main",
    notifier: BaseNotifier | Callable[[], BaseNotifier] | None = None,
    init_callbacks: List[CallbackFn] | None = None,
    post_callbacks: List[CallbackFn] | None = None,
) -> Callable[P, None]:  # type: ignore
    """
    Parameters
    ----------

    func:
        Function to execute.
    name : str, optional
        The name of the task. Default is "main".
    notifier : BaseNotifier | Callable[[], BaseNotifier] | None, optional
        The notifier object or callable that returns a notifier object. Default is None.
    init_callbacks : List[CallbackFn] | None, optional
        The list of callback functions to be executed during task initialization. Default is None.
    post_callbacks : List[CallbackFn] | None, optional
        The list of callback functions to be executed after the task completes. Default is None.

    Returns
    -------
    Callable[[_TaskFn], None]
        The decorator function that registers the task.
    """

    if name is _TASK_STORE:
        raise RuntimeError(f"Task with name {name} is already registered.")

    from absl_extra.callbacks import DEFAULT_INIT_CALLBACKS, DEFAULT_POST_CALLBACK

    if isinstance(notifier, Callable):  # type: ignore
        notifier = notifier()  # type: ignore
    if notifier is None:
        notifier = LoggingNotifier()

    if init_callbacks is None:
        init_callbacks = DEFAULT_INIT_CALLBACKS  # type: ignore

    if post_callbacks is None:
        post_callbacks = DEFAULT_POST_CALLBACK  # type: ignore

    _TASK_STORE[name] = make_task_func(
        name=name,
        notifier=notifier,
        init_callbacks=init_callbacks,
        post_callbacks=post_callbacks,
    )(func)

    return _TASK_STORE[name]  # type: ignore


def run(argv: List[str] | None = None, task_flag: str = "task", **kwargs):
    """
    Parameters
    ----------
    argv:
        CLI args passed to absl.app.run
    task_flag:
        Name of the CLI flag used to identify which task to run.
    kwargs:
        Kwargs passed to entrypoint function.

    Returns
    -------

    """
    flags.DEFINE_string(task_flag, default="main", help="Name of the function to execute.")

    def select_main(_):
        task_name = getattr(FLAGS, task_flag)
        if task_name not in _TASK_STORE:
            raise RuntimeError(f"Unknown {task_flag} {task_name}, registered are {list(_TASK_STORE.keys())}")
        func = _TASK_STORE[task_name]
        func(**kwargs)

    app.run(select_main, argv=argv)
