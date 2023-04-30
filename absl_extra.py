from __future__ import annotations

import functools
import json
from importlib import util
from typing import Callable, NamedTuple, TypeVar, Mapping
from functools import wraps
from absl import app, flags, logging
import inspect

T = TypeVar("T", bound=Callable, covariant=True)


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
    def notify_job_started(self, name: str):
        logging.info(f"Job {name} started.")

    def notify_job_finished(self, name: str):
        logging.info(f"Job {name} finished.")

    def notify_job_failed(self, name: str, exception: Exception):
        logging.fatal(f"Job {name} failed", exc_info=exception)


if util.find_spec("slack_sdk"):
    import slack_sdk

    class SlackNotifier(Notifier):
        def __init__(self, slack_token: str, channel_id: str):
            self.slack_token = slack_token
            self.channel_id = channel_id

        def notify_job_started(self, name: str):
            slack_client = slack_sdk.WebClient(token=self.slack_token)
            slack_client.chat_postMessage(
                channel=self.channel_id,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f" :ballot_box_with_check: Job {name} started.",
                        },
                    }
                ],
                text="Job Started!",
            )

        def notify_job_finished(self, name: str):
            slack_client = slack_sdk.WebClient(token=self.slack_token)
            slack_client.chat_postMessage(
                channel=self.channel_id,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":white_check_mark: Job {name} finished execution.",
                        },
                    }
                ],
                text="Job Finished!",
            )

        def notify_job_failed(self, name: str, exception: Exception):
            slack_client = slack_sdk.WebClient(token=self.slack_token)
            slack_client.chat_postMessage(
                channel=self.channel_id,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":x: Job {name} failed, reason:\n ```{exception}```",
                        },
                    }
                ],
                text="Job Finished!",
            )

else:
    logging.warning("slack_sdk not installed.")


class ExceptionHandlerImpl(app.ExceptionHandler):
    def __init__(self, name: str, notifier: Notifier):
        self.name = name
        self.notifier = notifier

    def handle(self, exception: Exception):
        self.notifier.notify_job_failed(self.name, exception)


def hook_main(
    main: Callable,
    *,
    app_name: str = "absl_app",
    notifier: Notifier | None = None,
    config_file: str | None = None,
    mongo_config: MongoConfig | Mapping[str, ...] | None = None,
) -> Callable:
    main_kwargs = {}

    if notifier is None:
        notifier = Notifier()
    if util.find_spec("ml_collections") and config_file is not None:
        config = config_flags.DEFINE_config_file("config")
        main_kwargs["config"] = config.value
    else:
        config = None

    if util.find_spec("pymongo") and mongo_config is not None:
        if isinstance(mongo_config, Mapping):
            mongo_config = MongoConfig(**mongo_config)
        db = MongoClient(mongo_config.uri).get_database(mongo_config.db_name)
        if mongo_config.collection is not None:
            db = db.get_collection(mongo_config.collection)
        main_kwargs["db"] = db

    def init_callback():
        logging.info("-" * 50)
        logging.info(
            f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
        )
        if config is not None:
            logging.info(
                f"Config: {json.dumps(config.value, sort_keys=True, indent=4)}"
            )
        logging.info("-" * 50)

        notifier.notify_job_started(app_name)

    app.install_exception_handler(ExceptionHandlerImpl(app_name, notifier))
    app.call_after_init(init_callback)

    @wraps(main)
    def wrapper(*args, **kwargs):
        ret_val = main(*args, **kwargs)
        notifier.notify_job_finished(app_name)
        return ret_val

    return functools.partial(wrapper, **main_kwargs)


def log_before(func: T, logger=logging.debug) -> T:
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        logger(
            f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


def log_after(func: T, logger=logging.debug) -> T:
    @wraps(func)
    def wrapper(*args, **kwargs):
        retval = func(*args, **kwargs)
        logger("Exited " + func.__name__ + "() with value: " + repr(retval))
        return retval

    return wrapper
