import datetime
import sys
import time

#
# def print_log_line(script, function, log_line, level=0, log_type='info'):
#     if log_type == 'error':
#         file_pointer = sys.stderr
#     else:
#         file_pointer = sys.stdout
#
#     if isinstance(log_line, list):
#         log_line = ', '.join(log_line)
#
#     tab_level = '\t' * level
#
#     timestamp = datetime.datetime.now()
#     print(timestamp.strftime('%Y-%m-%d %H:%M:%S'), f'{tab_level}{script}: {function}', log_line , file=file_pointer, flush=True)
#     return time.time()


# cedar_utils/log_utils.py
import logging
import os
import sys


def get_log_level(env_var: str = "CEDAR_LOG_LEVEL", default: str = "INFO") -> int:
    """Map env var to a logging level, with a safe default."""
    name = os.getenv(env_var, default).upper()
    return getattr(logging, name, logging.INFO)


class StdoutFilter(logging.Filter):
    """Allow only records below ERROR (DEBUG/INFO/WARNING)."""
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return record.levelno < logging.ERROR


class StderrFilter(logging.Filter):
    """Allow only ERROR and CRITICAL."""
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return record.levelno >= logging.ERROR


def setup_logging(
    env_var: str = "CEDAR_LOG_LEVEL",
    default_level: str = "INFO",
    force: bool = False,
) -> None:
    """
    Configure root logger once:
      - level from env
      - stdout for < ERROR
      - stderr for >= ERROR
    """
    root = logging.getLogger()

    if root.handlers and not force:
        # Already configured; don't double-add handlers
        return

    if force:
        # Strip any existing handlers (Jupyter, etc.)
        for h in root.handlers[:]:
            root.removeHandler(h)

    level = get_log_level(env_var, default_level)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler for DEBUG/INFO/WARNING
    h_out = logging.StreamHandler(stream=sys.stdout)
    h_out.setLevel(level)
    h_out.addFilter(StdoutFilter())
    h_out.setFormatter(formatter)

    # stderr handler for ERROR/CRITICAL
    h_err = logging.StreamHandler(stream=sys.stderr)
    h_err.setLevel(level)
    h_err.addFilter(StderrFilter())
    h_err.setFormatter(formatter)

    root.addHandler(h_out)
    root.addHandler(h_err)


def log_line(
    logger: logging.Logger,
    log_line,
    *,
    indent: int = 0,
    log_type: str = "info",
) -> None:
    """
    Drop-in-ish replacement for your print_log_line, but via logging.

    logger   : logger instance (module- or class-level)
    log_line : str or list
    indent   : indentation level (tabs)
    log_type : 'debug', 'info', 'warning', 'error', 'critical'
    """
    if isinstance(log_line, list):
        log_line = ", ".join(map(str, log_line))

    prefix = "\t" * indent
    msg = f"{prefix}{log_line}"

    log_method = getattr(logger, log_type, logger.info)
    log_method(msg)
