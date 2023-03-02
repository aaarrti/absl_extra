from typing import List, Optional, Callable
from absl import app
from ml_collections import config_flags, ConfigDict


from .notifiers import Notifier
from .utils import value_or_default, map_optional


MainFn = Callable[[List[str], Optional[ConfigDict]], None]


class ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, cmd: str, notifier: Notifier):
        self.cmd = cmd
        self.notifier = notifier

    def handle(self, exc: Exception):
        self.notifier.notify_job_failed(self.cmd, exc)


class App:
    def __init__(
        self, notifier: Optional[Notifier] = None, *, config_file: Optional[str] = None
    ):
        self.notifier = value_or_default(notifier, lambda: Notifier())
        self.config = map_optional(
            config_file, lambda v: config_flags.DEFINE_config_file("config", default=v)
        )

    def run(self, main: MainFn):
        def hook_main(argv):
            self.notifier.notify_job_started(argv[0])
            ex_handler = ExceptionHandlerImpl(argv[0], self.notifier)
            app.install_exception_handler(ex_handler)
            main(argv, map_optional(self.config, lambda v: v.value))
            self.notifier.notify_job_finished(argv[0])

        app.run(hook_main)
