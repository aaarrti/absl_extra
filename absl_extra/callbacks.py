from __future__ import annotations

from typing import Protocol, TYPE_CHECKING, type_check_only
from importlib import util
from absl import logging, flags
import json

if util.find_spec("ml_collections"):
    from ml_collections import ConfigDict
else:
    ConfigDict = None
if util.find_spec("pymongo"):
    from pymongo.collection import Collection
else:
    Collection = None

if TYPE_CHECKING:
    from absl_extra.notifier import BaseNotifier


@type_check_only
class CallbackFn(Protocol):
    def __call__(
        self,
        name: str,
        *,
        notifier: BaseNotifier,
        config: ConfigDict = None,
        db: Collection = None,
    ) -> None:
        ...


def log_absl_flags_callback(*args, **kwargs):
    logging.info("-" * 50)
    flags_dict = flags.FLAGS.flag_values_dict()
    if "config" in flags_dict:
        flags_dict["config"] = flags_dict["config"].to_dict()
    logging.info(f"ABSL flags: {json.dumps(flags_dict, sort_keys=True, indent=4)}")


def log_startup_callback(name: str, *, notifier: BaseNotifier, **kwargs):
    notifier.notify_task_started(name)


def log_shutdown_callback(name: str, *, notifier: BaseNotifier, **kwargs):
    notifier.notify_task_finished(name)
