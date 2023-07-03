from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import util
from typing import Callable

from absl import logging


class BaseNotifier(ABC):
    @abstractmethod
    def notify_job_started(self, name: str):
        raise NotImplementedError

    @abstractmethod
    def notify_job_finished(self, name: str):
        raise NotImplementedError

    @abstractmethod
    def notify_job_failed(self, name: str, exception: Exception):
        raise NotImplementedError


class NoOpNotifier(BaseNotifier):
    def notify_job_started(self, name: str):
        pass

    def notify_job_finished(self, name: str):
        pass

    def notify_job_failed(self, name: str, exception: Exception):
        raise NotImplementedError


class LoggingNotifier(BaseNotifier):

    def notify_job_started(self, name: str):
        logging.info(f"Job {name} started.")

    def notify_job_finished(self, name: str):
        logging.info(f"Job {name} finished.")

    def notify_job_failed(self, name: str, exception: Exception):
        logging.error(f"Job {name} failed with {exception}")


if util.find_spec("slack_sdk"):
    import slack_sdk

    class SlackNotifier(BaseNotifier):
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
                text="Job Failed!",
            )

else:
    logging.warning("slack_sdk not installed.")
