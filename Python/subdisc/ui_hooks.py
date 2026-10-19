"""Places where the java core code pops up a Swing dialog.

The core modules call these functions, the Qt gui replaces them at start
up (see gui/app.py). Without a gui they just log, or return None, so the
library and the xml autorun work headless.
"""

from . import log

HEADLESS = True


def _default_save(file_type=None, title=None):
    return None


def _default_open(file_type=None, title=None):
    return None


def _default_message(title, text, kind="info"):
    log.log_command_line("%s: %s" % (title, text))


def _default_confirm(title, text):
    log.log_command_line("%s: %s (assuming yes)" % (title, text))
    return True


def _default_input(title, text, default=""):
    return None


choose_save_file = _default_save
choose_open_file = _default_open
show_message = _default_message
confirm = _default_confirm
ask_input = _default_input
# set by the gui: show_result_window(table, selection, subgroup_discovery)
show_result_window = None
# set by the gui: show_cauc_window(target_column, statistics)
show_cauc_window = None
# set by the gui: choose_cui_domain(list_of_paths) -> path or None
choose_cui_domain = None
# set by the gui: choose_identifier(columns) -> (column_index, 'entrez'|'go') or None
choose_identifier = None


def install(save=None, open_=None, message=None, confirm_=None, input_=None):
    global choose_save_file, choose_open_file, show_message, confirm, ask_input, HEADLESS
    if save:
        choose_save_file = save
    if open_:
        choose_open_file = open_
    if message:
        show_message = message
    if confirm_:
        confirm = confirm_
    if input_:
        ask_input = input_
    HEADLESS = False
