import logging
from typing import Optional

from rich.logging import RichHandler


def use_rich_for_logging() -> None:
    """Use Rich as the main logger."""
    logging.basicConfig(
        format="%(message)s",  # noqa: WPS323
        datefmt="[%X]",  # noqa: WPS323
        handlers=[RichHandler(markup=True, rich_tracebacks=True, tracebacks_show_locals=True)],
    )


# Just returns a logger. Useful and important if rich hasn't been called yet.
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return Rich logger."""
    return logging.getLogger(name)
