from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import util

from absl import logging


class BaseNotifier(ABC):
    @abstractmethod
    def notify_task_started(self, name: str):
        raise NotImplementedError

    @abstractmethod
    def notify_task_finished(self, name: str):
        raise NotImplementedError

    @abstractmethod
    def notify_task_failed(self, name: str, exception: Exception):
        raise NotImplementedError


class NoOpNotifier(BaseNotifier):
    def notify_task_started(self, name: str):
        pass

    def notify_task_finished(self, name: str):
        pass

    def notify_task_failed(self, name: str, exception: Exception):
        pass


class LoggingNotifier(BaseNotifier):
    def notify_task_started(self, name: str):
        logging.info("-" * 50)
        logging.info(f"Task {name} started.")

    def notify_task_finished(self, name: str):
        logging.info("-" * 50)
        logging.info(f"Task {name} finished.")

    def notify_task_failed(self, name: str, exception: Exception):
        # TODO: walk the stacktrace and log arguments of failed call.
        logging.info("-" * 50)
        logging.error(f"Task {name} failed with {exception}")


if util.find_spec("slack_sdk"):
    import slack_sdk

    class SlackNotifier(BaseNotifier):
        def __init__(self, slack_token: str, channel_id: str):
            self.slack_token = slack_token
            self.channel_id = channel_id

        def notify_task_started(self, name: str):
            slack_client = slack_sdk.WebClient(token=self.slack_token)
            slack_client.chat_postMessage(
                channel=self.channel_id,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f" :ballot_box_with_check: Task {name} started.",
                        },
                    }
                ],
                text="Task Started!",
            )

        def notify_task_finished(self, name: str):
            slack_client = slack_sdk.WebClient(token=self.slack_token)
            slack_client.chat_postMessage(
                channel=self.channel_id,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":white_check_mark: Task {name} finished execution.",
                        },
                    }
                ],
                text="Task Finished!",
            )

        def notify_task_failed(self, name: str, exception: Exception):
            slack_client = slack_sdk.WebClient(token=self.slack_token)
            slack_client.chat_postMessage(
                channel=self.channel_id,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":x: Task {name} failed, reason:\n ```{repr(exception)}```",
                        },
                    }
                ],
                text="Task Failed!",
            )

else:
    logging.warning("slack_sdk not installed.")
