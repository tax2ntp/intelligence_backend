import datetime
import logging
from logging import Formatter, Logger, StreamHandler
from typing import Dict, Optional

import pytz


class ColorCode:
    """ANSI Color codes for terminal output"""

    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    CYAN = "\x1b[36;20m"
    MAGENTA = "\x1b[35;20m"
    RESET = "\x1b[0m"


class ThaiColorFormatter(Formatter):
    """Custom formatter with Thai time and colored output"""

    COLORS: Dict[int, str] = {
        logging.DEBUG: ColorCode.GREY,
        logging.INFO: ColorCode.GREEN,
        logging.WARNING: ColorCode.YELLOW,
        logging.ERROR: ColorCode.RED,
        logging.CRITICAL: ColorCode.BOLD_RED,
    }

    def __init__(self, fmt: str) -> None:
        super().__init__()
        self.fmt: str = fmt

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        thai_tz: pytz.timezone = pytz.timezone("Asia/Bangkok")
        thai_time: datetime.datetime = datetime.datetime.fromtimestamp(record.created, thai_tz)

        return f"{ColorCode.CYAN} {thai_time.strftime("%A")} {thai_time.day} {thai_time.month} {thai_time.year} {thai_time.strftime('%H:%M:%S')}{ColorCode.RESET}"

    def format(self, record: logging.LogRecord) -> str:
        color: str = self.COLORS.get(record.levelno, ColorCode.GREY)

        log_msg: str = self.fmt % {
            "asctime": self.formatTime(record),
            "levelname": f"{color}{record.levelname:8}{ColorCode.RESET}",
            "message": record.getMessage(),
        }

        return log_msg


class LoggerConfig:
    def __init__(self, name: str = "") -> None:
        self._logger: Logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        formatter: ThaiColorFormatter = ThaiColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler: StreamHandler = StreamHandler()
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def critical(self, message: str) -> None:
        self._logger.critical(message)

    @property
    def logger(self) -> Logger:
        return self._logger


logger: Logger = LoggerConfig().logger
