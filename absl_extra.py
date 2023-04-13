from __future__ import annotations

import json
from importlib import util
from typing import Callable, NamedTuple


from absl import app, flags, logging


if util.find_spec("pymongo"):
    from pymongo import MongoClient


if util.find_spec("ml_collections"):
    from ml_collections import config_flags


class MongoConfig(NamedTuple):
    uri: str
    db_name: str
    collection: str


class Notifier:
    def notify_job_started(self, cmd: str):
        logging.info(f"Job {cmd} started.")

    def notify_job_finished(self, cmd: str):
        logging.info(f"Job {cmd} finished.")

    def notify_job_failed(self, cmd: str, ex: Exception):
        logging.fatal(f"Job {cmd} failed", exc_info=ex)


if util.find_spec("slack_sdk"):
    import slack_sdk

    class SlackNotifier(Notifier):
        def __init__(self, slack_token: str, channel_id: str):
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
        app_name: str | None = None,
        notifier: Notifier | None = None,
        config_file: str | None = None,
        mongo_config: MongoConfig | None = None,
    ):
        if app_name is not None:
            self.app_name = app_name
        if notifier is None:
            notifier = Notifier()
        self.notifier = notifier
        if config_file is not None:
            self.config = config_flags.DEFINE_config_file("config")
        if mongo_config is not None:
            self.db = (
                MongoClient(mongo_config.uri)
                .get_database(mongo_config.db_name)
                .get_collection(mongo_config.collection)
            )

    def run(self, main: Callable):
        if hasattr(self, "app_name"):
            app_name = self.app_name
        else:
            app_name = "app"
        ex_handler = ExceptionHandlerImpl(app_name, self.notifier)
        app.install_exception_handler(ex_handler)
        kwargs = {}
        if hasattr(self, "config"):
            logging.info(
                f"Config: {json.dumps(self.config.value, sort_keys=True, indent=4)}"
            )
            logging.info("-" * 50)
            kwargs["config"] = self.config.value
        if hasattr(self, "db"):
            kwargs["db"] = self.db

        logging.info("-" * 50)
        logging.info(
            f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
        )
        logging.info("-" * 50)

        self.notifier.notify_job_started(app_name)
        app.run(main)
        self.notifier.notify_job_finished(app_name)
