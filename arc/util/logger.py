import logging
import os
import pprint
import sys
from typing import Optional


# ANSI codes that will generate colored text
def_color_map = {
    # reset should suffix any use of the other following prefixes
    "reset": "\x1b[0m",
    "brightred": "\x1b[31m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "light_blue": "\x1b[94m",
    "darkgreen": "\x1b[92m",
    "purple": "\x1b[95m",
}

def_level_color = {
    "CRITICAL": "brightred",
    "ERROR": "red",
    "WARNING": "yellow",
    "INFO": "darkgreen",
    "DEBUG": None,
    "TRACE": None,
}

# We never truly want infinite output, 1000 lines is probably enough
config = {
    "DEBUG": {"max_lines": 1000},
    "INFO": {"max_lines": 10},
    "default": {"max_lines": 50},
}

formats = {
    "bare": "{msg}",
    "level": "{level_str}| {msg}",
    "name": "{level_str}| {name} | {msg}",
    "full": "{level_str}| {name} | {funcName}:{lineno} - {msg}",
}


# The default schema uses simpler logging for info-level logs, as these are
# intended for consumption at all times (non-debugging)
styles = {
    "default": {"INFO": "name", "DEBUG": "name", "default": "full"},
}


class FancyFormatter(logging.Formatter):
    """Adds colors and structure to a log output"""

    def __init__(self, style: dict[str, str]):
        # The schema will have level-dependent format strings
        self.style = style

        # Level colors can also be overridden by changing the dictionary at the top
        self.level_color = def_level_color
        self.color_map = def_color_map

    def color_text(self, text: str, color: Optional[str] = None) -> str:
        """Wraps text in a color code"""
        if not color:
            return text
        prefix = self.color_map.get(color, "")
        if not prefix:
            logging.warning(f"{color} not defined in color_map")
        suffix = self.color_map["reset"] if prefix else ""
        return f"{prefix}{text}{suffix}"

    def level_fmt(self, level: str) -> str:
        return self.color_text(f"{level: <8}", self.level_color.get(level))

    def format(self, record: logging.LogRecord) -> str:
        """Automatically called when logging a record"""
        # Allows modification of the record attributes
        record_dict = vars(record)

        # Prepare a pretty version of the message
        curr_conf = config.get(record.levelname, config["default"])
        pretty = pprint.pformat(record_dict["msg"]).strip("'\"")
        total_lines = pretty.count("\n")
        if total_lines > curr_conf["max_lines"]:
            lines = pretty.splitlines()
            # TODO Abstract away the lines left after truncation (e.g. the 2's and 4)
            trunc = total_lines - 6
            pretty = "\n".join(lines[:3] + [f"...truncated {trunc} lines"] + lines[-3:])
        record_dict["level_str"] = self.level_fmt(record.levelname)
        record_dict["msg"] = pretty

        # Shortcut characters for adding extra color
        if record_dict["msg"].startswith("#!"):
            record_dict["msg"] = (
                self.color_text(record_dict["msg"][2:], "purple") + "\n"
            )
        else:
            record_dict["msg"] += "\n"

        # First use an indicated format from the log message, otherwise use the level-based
        # format indicated in the style indicated during Logger initialization.
        formatter = formats[
            record_dict.get("fmt")
            or self.style.get(record.levelname, self.style["default"])
        ]

        return formatter.format(**record_dict)


def fancy_logger(name: str, style: dict[str, str] = styles["default"], level=None):
    name_logger = logging.getLogger(name)
    name_logger.setLevel(level)
    name_logger.propagate = False

    # Make sure not to re-add the handlers if the same name is used
    if not name_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.terminator = ""
        handler.setFormatter(FancyFormatter(style=style))
        name_logger.addHandler(handler)

    return name_logger


if __name__ == "__main__":
    log = fancy_logger("test")
    log.setLevel(10)
    for level in ["trace", "debug", "info", "warning", "error"]:
        getattr(log, level)(f"This is a {level} test")
    log.info("#!Purple message")
    log.info(
        {
            "title": "This is a pretty dictionary",
            "reasons": ["indentation", "length-checking"],
            "strings": [str(i) * 8 for i in range(20)],
            # "codes": [str(i) * 8 for i in range(50)],
        }
    )
