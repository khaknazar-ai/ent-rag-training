"""Own the worker process tree before permitting child work to start."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from ctypes import wintypes
from typing import BinaryIO, Protocol, runtime_checkable

START_GATE_ENV = "RAG_WORKER_START_GATE"


class ChildTerminationError(OSError):
    """The owned process tree could not be confirmed stopped."""


class ChildStartCancelled(RuntimeError):
    """Stop arrived before work was permitted; owned cleanup has completed."""


@runtime_checkable
class _ProcessGroupAPI(Protocol):
    def killpg(self, pgid: int, sig: int, /) -> None: ...


@runtime_checkable
class _KillSignalAPI(Protocol):
    @property
    def SIGKILL(self) -> int: ...


def _require_process_groups(api: object) -> _ProcessGroupAPI:
    if not isinstance(api, _ProcessGroupAPI) or not callable(api.killpg):
        raise ChildTerminationError("POSIX process groups unavailable")
    return api


def _require_sigkill(api: object) -> int:
    if not isinstance(api, _KillSignalAPI):
        raise ChildTerminationError("POSIX kill signal unavailable")
    value = api.SIGKILL
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChildTerminationError("Invalid POSIX kill signal")
    return value


def _confirm_posix_group_exit(pid: int) -> None:
    """Fail closed if descendants/zombies still retain the owned group id."""

    deadline = time.monotonic() + 5
    while True:
        try:
            _require_process_groups(os).killpg(pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise ChildTerminationError(
                "Worker process group termination was not confirmed"
            )
        time.sleep(0.02)


class _JobAccounting(ctypes.Structure):
    _fields_ = [
        ("user_time", ctypes.c_longlong),
        ("kernel_time", ctypes.c_longlong),
        ("period_user_time", ctypes.c_longlong),
        ("period_kernel_time", ctypes.c_longlong),
        ("page_faults", ctypes.c_uint32),
        ("total", ctypes.c_uint32),
        ("active", ctypes.c_uint32),
        ("terminated", ctypes.c_uint32),
    ]


def windows_job_active(name: str) -> int:
    """Query only; never terminate processes from persisted identifiers."""
    from ctypes import wintypes as w

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenJobObjectW.argtypes = [w.DWORD, w.BOOL, w.LPCWSTR]
    api.OpenJobObjectW.restype = w.HANDLE
    api.QueryInformationJobObject.argtypes = [
        w.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        w.DWORD,
        ctypes.c_void_p,
    ]
    api.CloseHandle.argtypes = [w.HANDLE]
    handle = api.OpenJobObjectW(4, False, name)  # JOB_OBJECT_QUERY only.
    if not handle:
        error = ctypes.get_last_error()
        if error == 2:  # No named job survives once its last process/handle exits.
            return 0
        raise ctypes.WinError(error)
    try:
        accounting = _JobAccounting()
        if not api.QueryInformationJobObject(
            handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return accounting.active
    finally:
        api.CloseHandle(handle)


class _WindowsJob:
    def __init__(
        self, name: str | None = None, *, memory_bytes: int | None = None
    ) -> None:
        if memory_bytes is not None and (
            type(memory_bytes) is not int
            or not 64 * 1024 * 1024 <= memory_bytes <= 4 * 1024**3
        ):
            raise ValueError("Job memory limit must be 64 MiB..4 GiB")
        from ctypes import wintypes as w

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_longlong),
                ("job_time", ctypes.c_longlong),
                ("flags", w.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("process_limit", w.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", w.DWORD),
                ("scheduling", w.DWORD),
            ]

        class IOCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "read_ops",
                    "write_ops",
                    "other_ops",
                    "read_bytes",
                    "write_bytes",
                    "other_bytes",
                )
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", IOCounters),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        self._accounting = _JobAccounting
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for function_name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            (
                "SetInformationJobObject",
                [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD],
                w.BOOL,
            ),
            (
                "QueryInformationJobObject",
                [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p],
                w.BOOL,
            ),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("TerminateJobObject", [w.HANDLE, w.UINT], w.BOOL),
            ("OpenProcess", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            ("IsProcessInJob", [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
            ("GetExitCodeProcess", [w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
        ):
            function = getattr(self._api, function_name)
            function.argtypes = args
            function.restype = result
        self._compare = ctypes.WinDLL(
            "kernelbase", use_last_error=True
        ).CompareObjectHandles
        self._compare.argtypes = [w.HANDLE, w.HANDLE]
        self._compare.restype = w.BOOL
        self._health_handle = None
        self._health_pid: int | None = None
        ctypes.set_last_error(0)
        self._handle = self._api.CreateJobObjectW(None, name)
        error = ctypes.get_last_error()
        if not self._handle:
            raise ctypes.WinError(error)
        if name is not None and error == 183:  # Never adopt an existing named job.
            self._api.CloseHandle(self._handle)
            raise ChildTerminationError("Named containment job already exists")
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_bytes is not None:
            limits.basic.flags |= 0x0100 | 0x0200  # PROCESS_MEMORY | JOB_MEMORY
            limits.process_memory = memory_bytes
            limits.job_memory = memory_bytes
        if not self._api.SetInformationJobObject(
            self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            setup_error = ctypes.WinError(ctypes.get_last_error())
            self._api.CloseHandle(self._handle)
            raise setup_error

    def assign(self, pid: int) -> None:
        handle = self._api.OpenProcess(0x0101, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self._api.AssignProcessToJobObject(self._handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._api.CloseHandle(handle)

    def assign_handle(self, handle: int) -> None:
        """Assign a borrowed owned-process handle without closing or reopening it."""
        if not self._api.AssignProcessToJobObject(self._handle, handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def accepts_health_pid(self, pid: int) -> bool:
        """Pin a live reporter in this exact owned job, using query-only handles."""
        if not self._handle or type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
            return False
        if self._health_pid is not None and self._health_pid != pid:
            return False
        handle = self._api.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            member = wintypes.BOOL()
            code = wintypes.DWORD()
            if (
                not self._api.IsProcessInJob(handle, self._handle, ctypes.byref(member))
                or not member.value
                or not self._api.GetExitCodeProcess(handle, ctypes.byref(code))
                or code.value != 259  # STILL_ACTIVE
                or (
                    self._health_handle is not None
                    and not self._compare(self._health_handle, handle)
                )
            ):
                return False
            if self._health_handle is None:
                self._health_handle, self._health_pid = handle, pid
                handle = None  # Retain identity until owned job closure.
            return True
        finally:
            if handle:
                self._api.CloseHandle(handle)

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        try:
            if not self._api.TerminateJobObject(self._handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while True:
                accounting = self._accounting()
                if not self._api.QueryInformationJobObject(
                    self._handle,
                    1,
                    ctypes.byref(accounting),
                    ctypes.sizeof(accounting),
                    None,
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if accounting.active == 0:
                    return
                if time.monotonic() >= deadline:
                    raise ChildTerminationError("Worker job still has active processes")
                time.sleep(0.02)
        finally:
            if self._health_handle is not None:
                self._api.CloseHandle(self._health_handle)
                self._health_handle = None
            self._api.CloseHandle(self._handle)
            self._handle = None


class OwnedChild:
    """One gated child with inherited output and no unbounded pipe buffering."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        environment: Mapping[str, str] | None = None,
        environment_overlay: Mapping[str, str] | None = None,
        output: BinaryIO | None = None,
        start_cancelled: Callable[[], bool] | None = None,
        owner_job: _WindowsJob | None = None,
        gate_guard: Callable | None = None,
    ) -> None:
        self.command = command
        self.process: subprocess.Popen | None = None
        self._job: _WindowsJob | None = None
        self._environment = environment
        self._environment_overlay = environment_overlay or {}
        self._output = output
        self._start_cancelled = start_cancelled or (lambda: False)
        self._owner_job = owner_job
        self._gate_guard = gate_guard or nullcontext
        self.termination_confirmed = False

    @property
    def pid(self) -> int:
        assert self.process is not None
        return self.process.pid

    def __enter__(self) -> OwnedChild:
        if os.name == "nt":
            self._job = _WindowsJob()
        try:
            if self._start_cancelled():
                raise ChildStartCancelled("Owned child start cancelled")
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                env={
                    **(os.environ if self._environment is None else self._environment),
                    **self._environment_overlay,
                    START_GATE_ENV: "1",
                },
                stdout=self._output,
                stderr=subprocess.STDOUT if self._output is not None else None,
                start_new_session=os.name != "nt",
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            if self._job is not None:
                self._job.assign(self.process.pid)
            if self._owner_job is not None:
                self._owner_job.assign(self.process.pid)
            if self._start_cancelled():
                raise ChildStartCancelled("Owned child start cancelled")
            assert self.process.stdin is not None
            with self._gate_guard():
                with suppress(BrokenPipeError):
                    self.process.stdin.write(b"1")
                    self.process.stdin.flush()
                with suppress(BrokenPipeError):
                    self.process.stdin.close()
            return self
        except BaseException:
            self.close()
            raise

    def poll(self) -> int | None:
        assert self.process is not None
        return self.process.poll()

    def accepts_health_pid(self, pid: int) -> bool:
        """Allow a Windows venv interpreter inside this child's containment job."""
        return bool(
            os.name == "nt"
            and self.process is not None
            and self.process.poll() is None
            and self._job is not None
            and self._job.accepts_health_pid(pid)
        )

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.termination_confirmed:
            return
        try:
            if self._job is not None:
                self._job.close()
                self._job = None
            elif self.process is not None and os.name != "nt":
                with suppress(ProcessLookupError):
                    _require_process_groups(os).killpg(
                        self.process.pid, _require_sigkill(signal)
                    )
        finally:
            if self.process is not None:
                try:
                    if self.process.poll() is None:
                        self.process.kill()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired as exc:
                        raise ChildTerminationError(
                            "Worker termination was not confirmed"
                        ) from exc
                    if os.name != "nt":
                        _confirm_posix_group_exit(self.process.pid)
                finally:
                    if self.process.stdin is not None:
                        self.process.stdin.close()
        self.termination_confirmed = True
