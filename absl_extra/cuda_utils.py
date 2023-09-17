from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import util
from typing import List

from absl import logging
from toolz.dicttoolz import valmap
import sys

if sys.version_info >= (3, 10):
    dc_kw = dict(slots=True)
else:
    dc_kw = dict()


@dataclass(frozen=True, **dc_kw)
class MemoryInfo:
    """All values are in GB."""

    total: float
    free: float
    used: float

    def __repr__(self) -> str:
        return repr(valmap(lambda v: f"{v} GB", asdict(self)))


if util.find_spec("pynvml") is not None:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetCudaComputeCapability,
        nvmlDeviceGetHandleByIndex,
    )

    class NvmlState:
        def __init__(self):
            self.initialized = False

        def maybe_init(self):
            if not self.initialized:
                nvmlInit()
                self.initialized = True

        def __del__(self):
            if self.initialized:
                nvmlShutdown()

    nvm_state = NvmlState()

    def cuda_devices_available() -> bool:
        nvm_state.maybe_init()
        deviceCount = nvmlDeviceGetCount()
        return deviceCount > 0

    def supports_mixed_precision() -> bool:
        """
        Checks if CUDA devices support mixed float16 precision.

        Returns
        -------

        bool:
            True, if all devices have `Compute Capability` of 7.5 or higher.
            False, if there are no CUDA devices.

        """
        nvm_state.maybe_init()
        deviceCount = nvmlDeviceGetCount()

        if deviceCount == 0:
            logging.warning("No CUDA devices found, mixed f16 -> NOT OK.")
            return False

        mixed_f16_ok = None

        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            cc = nvmlDeviceGetCudaComputeCapability(handle)
            name = nvmlDeviceGetName(handle)
            cc = float(f"{cc[0]}.{cc[1]}")

            if cc >= 7.5:
                logging.info(f"{name} has CC {cc} (>= 7.5) -> mixed float16 OK.")
                if mixed_f16_ok is None:
                    mixed_f16_ok = True
                else:
                    mixed_f16_ok = mixed_f16_ok and True
            else:
                logging.info(f"{name} has CC {cc} (< 7.5) -> mixed float16 NOT OK.")
                mixed_f16_ok = False

        return bool(mixed_f16_ok)

    def get_memory_info() -> List[MemoryInfo]:
        """
        Get memory info for CUDA devices

        Parameters
        ----------

        Returns
        -------

        memory_info:
            List of total, free, used memory for each CUDA device in GB`.
            Empty list is there are no CUDA devices.

        """
        nvm_state.maybe_init()
        deviceCount = nvmlDeviceGetCount()

        memory_consumption_list = []

        if deviceCount == 0:
            logging.error("No CUDA devices found.")
            return []

        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory = nvmlDeviceGetMemoryInfo(handle)
            memory_info = MemoryInfo(
                used=bytes_to_gb(memory.used),
                total=bytes_to_gb(memory.total),
                free=bytes_to_gb(memory.free),
            )
            memory_consumption_list.append(memory_info)

        return memory_consumption_list

else:

    def supports_mixed_precision() -> bool:
        logging.error("nvidia-ml-py not installed")
        return False

    def get_memory_info() -> List[MemoryInfo]:
        logging.error("nvidia-ml-py not installed")
        return []

    def cuda_devices_available() -> bool:
        logging.error("nvidia-ml-py not installed")
        return False


def bytes_to_gb(value: int) -> float:
    return value / (1024**3)
