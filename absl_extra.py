from __future__ import annotations

import json
from importlib import util
from typing import Callable, NamedTuple
from functools import partial, wraps
from absl import app, flags, logging


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

else:
    logging.warning("slack_sdk not installed.")


class ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, cmd: str, notifier: Notifier):
        self.cmd = cmd
        self.notifier = notifier

    def handle(self, exc: Exception):
        self.notifier.notify_job_failed(self.cmd, exc)


def hook_main(
    main: Callable,
    app_name: str | None = None,
    notifier: Notifier | None = None,
    config_file: str | None = None,
    mongo_config: MongoConfig | None = None,
):
    if notifier is None:
        notifier = Notifier()
    if util.find_spec("ml_collections") and config_file is not None:
        config = config_flags.DEFINE_config_file("config")
    else:
        config = None
    if util.find_spec("pymongo") and mongo_config is not None:
        db = MongoClient(mongo_config.uri).get_database(mongo_config.db_name)
        if mongo_config.collection is not None:
            db = db.get_collection(mongo_config.collection)
    else:
        db = None

    @wraps(main)
    def wrapper(cmd: str):
        if app_name is None:
            _app_name = cmd
        else:
            _app_name = app_name

        app.install_exception_handler(ExceptionHandlerImpl(cmd, notifier))

        kwargs = {}
        if config is not None:
            logging.info(
                f"Config: {json.dumps(config.value, sort_keys=True, indent=4)}"
            )
            logging.info("-" * 50)
            kwargs["config"] = config.value
        if db is not None:
            kwargs["db"] = db

        logging.info("-" * 50)
        logging.info(
            f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
        )
        logging.info("-" * 50)

        notifier.notify_job_started(app_name)
        app.run(partial(main, **kwargs))
        notifier.notify_job_finished(app_name)

    return wrapper
