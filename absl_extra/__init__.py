from __future__ import annotations
from importlib import util


def is_lib_installed(name: str) -> bool:
    return util.find_spec(name) is not None


from absl_extra.src.tasks import run, register_task
from absl_extra.src.notifier import BaseNotifier

if is_lib_installed("slack_sdk"):
    from absl_extra.src.notifier import SlackNotifier
from absl_extra.src.logging_utils import (
    log_before,
    log_after,
    setup_logging,
    log_exception,
)

if is_lib_installed("pymongo"):
    from absl_extra.src.tasks import MongoConfig
if is_lib_installed("tensorflow"):
    from absl_extra.src import tf_utils

if is_lib_installed("jax"):
    from absl_extra.src import jax_utils
