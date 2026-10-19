"""Logging, same switches as the old Log class.

Most of the program only ever calls log_command_line(), the file streams
are kept for completeness (they write into ./log/ when opened).
"""

import os
import sys
import time

DEBUG = True
ERROR = True
SQLLOG = False
SQLRESULTLOG = False
REFINEMENTLOG = False
COMMANDLINELOG = True
LOG = True
FORCECOMMANDLINELOG = False

LOGPATH = "./log/"

_debug_stream = sys.stdout
_error_stream = sys.stderr
_log_stream = sys.stdout
_sql_log_stream = sys.stdout
_refinement_log_stream = sys.stdout

# the gui can hook in here to show messages as well
_listeners = []


def add_listener(fn):
    if fn not in _listeners:
        _listeners.append(fn)


def remove_listener(fn):
    if fn in _listeners:
        _listeners.remove(fn)


def open_file_output_streams():
    global _debug_stream, _error_stream, _log_stream, _sql_log_stream, _refinement_log_stream
    if not (SQLLOG or REFINEMENTLOG or ERROR or LOG or DEBUG):
        return
    try:
        os.makedirs(LOGPATH, exist_ok=True)
        log_command_line("creating files in directory: " + LOGPATH)
        if DEBUG:
            _debug_stream = open(LOGPATH + "debug.wri", "w", encoding="utf-8")
        if ERROR:
            _error_stream = open(LOGPATH + "error.wri", "w", encoding="utf-8")
        if LOG:
            _log_stream = open(LOGPATH + "log.wri", "w", encoding="utf-8")
        if SQLLOG:
            _sql_log_stream = open(LOGPATH + "sql.wri", "w", encoding="utf-8")
        if REFINEMENTLOG:
            _refinement_log_stream = open(LOGPATH + "refinement.wri", "w", encoding="utf-8")
    except OSError as ex:
        error(str(ex))


def close_file_output_streams():
    for s in (_debug_stream, _log_stream, _error_stream, _sql_log_stream, _refinement_log_stream):
        try:
            s.flush()
            if s not in (sys.stdout, sys.stderr):
                s.close()
        except Exception:
            pass


def error(s):
    try:
        _error_stream.write(s + "\n")
    except Exception:
        pass


def debug(s):
    # disabled in the original as well
    pass


def log_sql(s):
    if SQLLOG:
        try:
            _sql_log_stream.write(s + "\n")
        except Exception:
            pass
    log_command_line(s)


def log_refinement(s):
    if REFINEMENTLOG:
        try:
            _refinement_log_stream.write(s + "\n")
        except Exception:
            _log(s)


def log_command_line(s):
    if COMMANDLINELOG:
        try:
            print(s, flush=True)
        except Exception:
            # e.g. a console that can not print some character
            try:
                print(str(s).encode("ascii", "replace").decode("ascii"), flush=True)
            except Exception:
                pass
    for fn in list(_listeners):
        try:
            fn(s)
        except Exception:
            pass


def to_unique_file(file_name, content):
    try:
        os.makedirs(LOGPATH, exist_ok=True)
        unique_id = str(int(time.time() * 1000))
        with open(LOGPATH + file_name + unique_id + ".wri", "w", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")
    except OSError as ex:
        error(str(ex))


def _log(s):
    if LOG:
        try:
            _log_stream.write(s + "\n")
        except Exception:
            pass
    elif FORCECOMMANDLINELOG:
        print("  " + s)
