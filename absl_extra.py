import json
from typing import Callable, List, Optional, TypeVar
from operator import attrgetter
from absl import app, flags, logging
from ml_collections import ConfigDict, config_flags
from functools import wraps
from dotenv import load_dotenv
import slack_sdk

MainFn = Callable[[List[str], Optional[ConfigDict]], None]

T = TypeVar("T")
R = TypeVar("R")


def value_or_default(value: T, default_factory: Callable[[], T]) -> T:
    if value is not None:
        return value

    return default_factory()


def map_optional(value: Optional[T], fn: Callable[[T], R]) -> Optional[R]:
    if value is None:
        return None
    return fn(value)


class Notifier:
    def notify_job_started(self, cmd: str):
        logging.info(f"Job {cmd} started.")

    def notify_job_finished(self, cmd: str):
        logging.info(f"Job {cmd} finished.")

    def notify_job_failed(self, cmd: str, ex: Exception):
        logging.fatal(f"Job {cmd} failed", exc_info=ex)


class SlackNotifier(Notifier):
    def __init__(
        self, slack_token: Optional[str] = None, channel_id: Optional[str] = None
    ):
        from os import environ

        if slack_token is None:
            if "SLACK_BOT_TOKEN" not in environ:
                raise ValueError("Provide slack token.")
            else:
                slack_token = environ["SLACK_BOT_TOKEN"]

        if channel_id is None:
            if "CHANNEL_ID" not in environ:
                raise ValueError("Provide channel id.")
            else:
                channel_id = environ["CHANNEL_ID"]

        self.slack_token = slack_token
        self.channel_id = channel_id

    def notify_job_started(self, cmd: str):
        slack_client = slack_sdk.WebClient(token=self.slack_token)
        slack_client.chat_postMessage(
            channel=self.channel_id,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f" :ballot_box_with_check: Job {cmd} started.",
                    },
                }
            ],
            text="Job Started!",
        )

    def notify_job_finished(self, cmd: str):
        slack_client = slack_sdk.WebClient(token=self.slack_token)
        slack_client.chat_postMessage(
            channel=self.channel_id,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: Job {cmd} finished execution.",
                    },
                }
            ],
            text="Job Finished!",
        )

    def notify_job_failed(self, cmd: str, ex: Exception):
        slack_client = slack_sdk.WebClient(token=self.slack_token)
        slack_client.chat_postMessage(
            channel=self.channel_id,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":x: Job {cmd} failed, reason:\n ```{ex}```",
                    },
                }
            ],
            text="Job Finished!",
        )
        raise ex


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
        env_file: Optional[str] = None,
    ):
        self.name = name
        self.notifier = value_or_default(notifier, lambda: Notifier())
        self.config = map_optional(
            config_file, lambda v: config_flags.DEFINE_config_file("config", default=v)
        )
        if env_file is not None:
            load_dotenv(env_file, verbose=True)

    def run(self, main: MainFn):
        @wraps(main)
        def hook_main(argv):
            app_name = value_or_default(self.name, lambda: argv[0])
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
            main(argv, map_optional(self.config, attrgetter("value")))
            self.notifier.notify_job_finished(app_name)

        app.run(hook_main)
