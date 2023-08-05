import sys
import dataclasses
import functools


if sys.version_info >= (3, 10):
    dataclass = functools.partial(
        dataclasses.dataclass, slots=True, frozen=True, kw_only=True
    )
else:
    dataclass = functools.partial(dataclasses.dataclass, frozen=True)
