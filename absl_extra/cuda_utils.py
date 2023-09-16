from __future__ import annotations

from typing import Literal, TYPE_CHECKING, overload, List, Protocol, no_type_check
from importlib import util
from threading import Lock
from dataclasses import dataclass, asdict

from absl import logging
from toolz.dicttoolz import valmap


if TYPE_CHECKING:

    class BytesMemory(Protocol):
        value: int
        unit: Literal["bytes"]

    class MBMemory(Protocol):
        value: int | float
        unit: Literal["MB"]

    class GBMemory(Protocol):
        value: int | float
        unit: Literal["MB"]

    class BytesMemoryInfo(Protocol):
        total: BytesMemory
        free: BytesMemory
        used: BytesMemory

    class MBMemoryInfo(Protocol):
        total: MBMemory
        free: MBMemory
        used: MBMemory

    class GBMemoryInfo(Protocol):
        total: GBMemory
        free: GBMemory
        used: GBMemory


@dataclass(frozen=True, slots=True, repr=False)
class MemoryType:
    value: float | int
    unit: Literal["bytes", "MB", "GB"] = "bytes"

    def __repr__(self) -> str:
        return str(self.value) if self.unit == "bytes" else f"{self.value} {self.unit}"

    @overload
    def cast(self, unit: Literal["GB"]) -> GBMemory:
        ...

    @overload
    def cast(self, unit: Literal["MB"]) -> MBMemory:
        ...

    @overload
    def cast(self, unit: Literal["bytes"]) -> BytesMemory:
        ...

    @no_type_check
    def cast(self, unit: Literal["bytes", "MB", "GB"]) -> MemoryType:
        if unit == self.unit:
            return self

        value = self.value
        # There is probably a normal way of implementing it, but I was to lazy so I just hardcoded it.
        if self.unit == "bytes":
            if unit == "MB":
                value /= 1024
            if unit == "GB":
                value /= 1024**2

        if self.unit == "MB":
            if unit == "bytes":
                value *= 1024
            if unit == "GB":
                value /= 1024

        if self.unit == "GB":
            if unit == "MB":
                value *= 1024
            if unit == "bytes":
                value *= 1024**2

        return MemoryType(value=value, unit=unit)


@dataclass(frozen=True, slots=True)
class MemoryInfo:
    total: MemoryType
    free: MemoryType
    used: MemoryType

    def __repr__(self) -> str:
        return repr(valmap(repr, asdict(self)))

    @overload
    def cast(self, unit: Literal["bytes"]) -> BytesMemoryInfo:
        ...

    @overload
    def cast(self, unit: Literal["MB"]) -> MBMemoryInfo:
        ...

    @overload
    def cast(self, unit: Literal["GB"]) -> GBMemoryInfo:
        ...

    @no_type_check
    def cast(self, unit: Literal["bytes", "MB", "GB"]) -> MemoryInfo:
        return MemoryInfo(total=self.total.cast(unit), free=self.free.cast(unit), used=self.used.cast(unit))


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
            self.lock = Lock()

        def maybe_init(self):
            with self.lock:
                if not self.initialized:
                    nvmlInit()
                    self.initialized = True

        def __del__(self):
            with self.lock:
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

    def get_memory_info(unit: Literal["bytes", "MB", "GB"] = "GB") -> List[MemoryInfo]:
        """
        Get memory info for CUDA devices

        Parameters
        ----------

        unit:
            Memory unit "bytes", "MB", "GB".

        Returns
        -------

        memory_info:
            List of total, free, used memory for each CUDA device in `unit`.
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
                used=MemoryType(value=memory.used),
                total=MemoryType(value=memory.total),
                free=MemoryType(value=memory.free),
            ).cast(unit)
            memory_consumption_list.append(memory_info)

        return memory_consumption_list  # type: ignore

else:

    def supports_mixed_precision() -> bool:
        logging.error("pynvml not installed")
        return False

    def get_memory_info(unit: Literal["bytes", "MB", "GB"] = "GB") -> List[MemoryInfo]:
        logging.error("pynvml not installed")
        return []

    def cuda_devices_available() -> bool:
        logging.error("pynvml not installed")
        return False
