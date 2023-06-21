from __future__ import annotations
from importlib import util


def is_lib_installed(name: str) -> bool:
    return util.find_spec(name) is not None


from absl_extra.tasks import run, register_task
from absl_extra.notifier import BaseNotifier

if is_lib_installed("slack_sdk"):
    from absl_extra.notifier import SlackNotifier
from absl_extra.logging_utils import log_before, log_after, setup_logging

if is_lib_installed("pymongo"):
    from absl_extra.tasks import MongoConfig
if is_lib_installed("tensorflow"):
    from absl_extra.tf_utils import (
        supports_mixed_precision,
        make_gpu_strategy,
        make_tpu_strategy,
        requires_gpu,
    )
