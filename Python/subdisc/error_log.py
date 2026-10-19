import inspect
import os

from . import log


class ErrorLog:
    # error descriptions
    ATTRIBUTE_TYPE_UNKNOWN = "AttributeType unknown"
    TARGET_TYPE_UNKNOWN = "TargetType unknown"
    FILE_NOT_FOUND = "File not found"
    IOEXCEPTION = "IOException"

    @staticmethod
    def get_error_description(ex):
        if ex is None:
            return ""
        if isinstance(ex, FileNotFoundError):
            return ErrorLog.FILE_NOT_FOUND
        if isinstance(ex, OSError):
            return ErrorLog.IOEXCEPTION
        return "Unknown Error"

    @staticmethod
    def log(path, ex):
        """Logs where the error happened, call it right after the error."""
        if ex is not None:
            an_error = " " + ErrorLog.get_error_description(ex)
            msg = str(ex) if str(ex) else ""
            if msg:
                msg += "/n"   # sic, same typo as the original
        else:
            an_error = ""
            msg = ""
        frame = inspect.stack()[1] if len(inspect.stack()) > 1 else None
        where = (frame.filename.rsplit(os.sep, 1)[-1] + "." + frame.function) if frame else "No StackTrace available."
        log.log_command_line("%s%s: '%s'.\n%s" % (where, an_error, "null" if path is None else os.path.abspath(path), msg))
