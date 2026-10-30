"""Bounded Windows file-handle release wait after owned process termination."""

import tempfile
import time


class OwnedTemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        deadline = time.monotonic() + 2
        while True:
            try:
                return super().cleanup()
            except PermissionError as error:
                if (
                    getattr(error, "winerror", None) != 32
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.02)
