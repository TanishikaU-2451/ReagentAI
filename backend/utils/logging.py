"""ReagentAI Centralized Logging.

Uses loguru for structured logging across all modules.
Tracks: agent reasoning, retrieval operations, execution results,
debug loops, and errors.
"""

import sys
from pathlib import Path

from loguru import logger

from backend.config.settings import settings


def setup_logging() -> None:
    """Configure centralized logging with loguru.

    Sets up file and console sinks with rotation and filtering.
    """
    log_dir = Path(settings.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console sink
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # General application log
    logger.add(
        str(log_dir / "reagentai.log"),
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # Agent-specific log
    logger.add(
        str(log_dir / "agents.log"),
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        filter=lambda record: "agent" in record["name"].lower(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    )

    # Execution log
    logger.add(
        str(log_dir / "execution.log"),
        level="INFO",
        rotation="50 MB",
        retention="30 days",
        filter=lambda record: "execution" in record["name"].lower(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )

    # Error-only log
    logger.add(
        str(log_dir / "errors.log"),
        level="ERROR",
        rotation="20 MB",
        retention="60 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    logger.info("ReagentAI logging initialized")


def get_logger(name: str):
    """Get a named logger instance.

    Args:
        name: Module or component name for log filtering.

    Returns:
        A loguru logger bound with the given name.
    """
    return logger.bind(name=name)
