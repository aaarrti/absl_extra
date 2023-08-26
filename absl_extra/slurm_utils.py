import signal
from types import FrameType


class JobCanceledException(RuntimeError):
    def __init__(self):
        super().__init__("Slurm canceled the job.")


def register_sigterm_handler():
    def handler(signum: signal.SIGTERM, frame: FrameType):
        raise JobCanceledException()

    signal.signal(signal.SIGTERM, handler)
