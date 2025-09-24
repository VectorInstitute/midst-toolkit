"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies.

https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

# TODO is this file necessary at all?

import datetime
import json
import os
import os.path as osp
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import IO, Any


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq: Iterable[str]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file: str | IO[str]):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            # ruff: noqa: SIM115
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), "expected file or str, got %s" % filename_or_file
            self.file = filename_or_file  # type: ignore[assignment]
            self.own_file = False

    def writekvs(self, kvs: dict[str, Any]) -> None:
        # Create strings for printing
        key2str = {}
        for key, val in sorted(kvs.items()):
            valstr = "%-8.3g" % val if hasattr(val, "__float__") else str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for key, val in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append("| %s%s | %s%s |" % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val))))
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s: str) -> str:
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq: Iterable[str]) -> None:
        seq = list(seq)
        for i, elem in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename: str):
        self.file = open(filename, "wt")
        # ruff: noqa: SIM115

    def writekvs(self, kvs: dict[str, Any]) -> None:
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename: str):
        self.file = open(filename, "w+t")
        # ruff: noqa: SIM115
        self.keys: list[str] = []
        self.sep = ","

    def writekvs(self, kvs: dict[str, Any]) -> None:
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for i, k in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for i, k in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """Dumps key/value pairs into TensorBoard's numeric format."""

    def __init__(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.core.util import event_pb2
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs: dict[str, Any]) -> None:
        def summary_val(k: str, v: Any) -> Any:
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self) -> None:
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format: str, ev_dir: str, log_suffix: str = "") -> KVWriter | SeqWriter:
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    if format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    if format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    if format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    if format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key: str, val: Any) -> None:
    """
    Log a value of some diagnostic.

    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key: str, val: Any) -> None:
    """The same as logkv(), but if called many times, values averaged."""
    get_current().logkv_mean(key, val)


def logkvs(d: dict[str, Any]) -> None:
    """Log a dictionary of key-value pairs."""
    for k, v in d.items():
        logkv(k, v)


def dumpkvs() -> dict[str, Any]:
    """Write all of the diagnostics from the current iteration."""
    return get_current().dumpkvs()


def getkvs() -> dict[str, Any]:
    return get_current().name2val


def log(*args: Iterable[Any], level: int = INFO) -> None:
    """
    Logs the args in the desired level.

    Write the sequence of args, with no separators, to the console and output
    files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args: Iterable[Any]) -> None:
    log(*args, level=DEBUG)


def info(*args: Iterable[Any]) -> None:
    log(*args, level=INFO)


def warn(*args: Iterable[Any]) -> None:
    log(*args, level=WARN)


def error(*args: Iterable[Any]) -> None:
    log(*args, level=ERROR)


def set_level(level: int) -> None:
    """Set logging threshold on current logger."""
    get_current().set_level(level)


def set_comm(comm: Any | None) -> None:
    get_current().set_comm(comm)


def get_dir() -> str:
    """
    Get directory that log files are being written to.

    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename: str) -> Generator[None, None, None]:
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart


def profile(n: str) -> Callable:
    """
    Usage.

    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):  # type: ignore
        def func_wrapper(*args, **kwargs):  # type: ignore
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir: str, output_formats: list[KVWriter | SeqWriter], comm: Any | None = None):
        # ruff: noqa: D107
        self.name2val: defaultdict[str, float] = defaultdict(float)  # values this iteration
        self.name2cnt: defaultdict[str, int] = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key: str, val: Any) -> None:
        # ruff: noqa: D102
        self.name2val[key] = val

    def logkv_mean(self, key: str, val: Any) -> None:
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self) -> dict[str, Any]:
        # ruff: noqa: D102
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(  # type: ignore[assignment]
                self.comm,
                {name: (val, self.name2cnt.get(name, 1)) for (name, val) in self.name2val.items()},
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args: Iterable[Any], level: int = INFO) -> None:
        # ruff: noqa: D102
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level: int) -> None:
        # ruff: noqa: D102
        self.level = level

    def set_comm(self, comm: Any | None) -> None:
        # ruff: noqa: D102
        self.comm = comm

    def get_dir(self) -> str:
        # ruff: noqa: D102
        return self.dir

    def close(self) -> None:
        # ruff: noqa: D102
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args: Iterable[Any]) -> None:
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def get_rank_without_mpi_import() -> int:
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def mpi_weighted_mean(comm: Any, local_name2valcount: dict[str, tuple[float, float]]) -> dict[str, float]:
    """
    Copied from below.

    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum: defaultdict[str, float] = defaultdict(float)
        name2count: defaultdict[str, float] = defaultdict(float)
        for n2vc in all_name2valcount:
            for name, (val, count) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn("WARNING: tried to compute mean on non-float {}={}".format(name, val))
                        # ruff: noqa: B028
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    return {}


def configure(
    dir: str | None = None,
    format_strs: list[str] | None = None,
    comm: Any | None = None,
    log_suffix: str = "",
) -> None:
    """If comm is provided, average all numerical stats across that comm."""
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs_filter = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs_filter]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)  # type: ignore[assignment]
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger() -> None:
    # ruff: noqa: D103
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset() -> None:
    # ruff: noqa: D103
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):  # type: ignore
    # ruff: noqa: D103
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        assert Logger.CURRENT is not None
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger


def get_current() -> Logger:
    # ruff: noqa: D103
    if Logger.CURRENT is None:
        _configure_default_logger()

    assert isinstance(Logger.CURRENT, Logger)
    return Logger.CURRENT
