from __future__ import annotations

import os
from typing import Optional

import slack_sdk
from absl import logging


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
        if slack_token is None:
            if "SLACK_BOT_TOKEN" not in os.environ:
                raise ValueError("Provide slack token.")
            else:
                slack_token = os.environ["SLACK_BOT_TOKEN"]

        if channel_id is None:
            if "CHANNEL_ID" not in os.environ:
                raise ValueError("Provide channel id.")
            else:
                channel_id = os.environ["CHANNEL_ID"]

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
