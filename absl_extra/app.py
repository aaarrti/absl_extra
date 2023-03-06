import json
from typing import Callable, List, Optional

from absl import app, flags, logging
from ml_collections import ConfigDict, config_flags

from .notifiers import Notifier
from .utils import map_optional, value_or_default

MainFn = Callable[[List[str], Optional[ConfigDict]], None]


class ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, cmd: str, notifier: Notifier):
        self.cmd = cmd
        self.notifier = notifier

    def handle(self, exc: Exception):
        self.notifier.notify_job_failed(self.cmd, exc)


class App:
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        notifier: Optional[Notifier] = None,
        config_file: Optional[str] = None,
    ):
        self.name = name
        self.notifier = value_or_default(notifier, lambda: Notifier())
        self.config = map_optional(
            config_file, lambda v: config_flags.DEFINE_config_file("config", default=v)
        )

    def run(self, main: MainFn):
        def hook_main(argv):
            app_name = value_or_default(self.name, argv[0])
            self.notifier.notify_job_started(app_name)
            ex_handler = ExceptionHandlerImpl(app_name, self.notifier)
            app.install_exception_handler(ex_handler)
            logging.info("-" * 50)
            logging.info(
                f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
            )
            logging.info("-" * 50)
            if self.config is not None:
                logging.info(
                    f"Config: {json.dumps(self.config.value, sort_keys=True, indent=4)}"
                )
                logging.info("-" * 50)

            main(argv, map_optional(self.config, lambda v: v.value))
            self.notifier.notify_job_finished(app_name)

        app.run(hook_main)
