"""Starting the gui (java SubDisc.main) and plugging the Qt dialogs into
ui_hooks, so the core code can ask for files and show messages.

All hooks may be called from the mining thread, they hop over to the gui
thread through common.bridge().
"""

import os
import sys
import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QFileDialog, QInputDialog, QMessageBox,
                               QSplashScreen)

from .. import ui_hooks
from .. import xml_auto_run
from ..attribute_type import AttributeType
from ..file_type import FileType
from . import common
from . import theme

_app = None


def _parent():
    return QApplication.activeWindow()


def _last_dir():
    from ..file_handler import FileHandler
    d = FileHandler.last_file_location
    return d if d and os.path.isdir(d) else os.getcwd()


def _remember(path):
    from ..file_handler import FileHandler
    if path:
        FileHandler.last_file_location = os.path.dirname(os.path.abspath(path))


def _filters(file_type):
    if file_type is None or file_type == FileType.ALL_DATA_FILES:
        return ";;".join([FileType.ALL_DATA_FILES.qt_filter(), FileType.TXT.qt_filter(),
                          FileType.ARFF.qt_filter(), FileType.XML.qt_filter(), "All Files (*)"])
    return file_type.qt_filter() + ";;All Files (*)"


def _open_dialog(file_type=None, title=None):
    path, _ = QFileDialog.getOpenFileName(_parent(), title or "Open", _last_dir(), _filters(file_type))
    _remember(path)
    return path or None


def _save_dialog(file_type=None, title=None):
    path, _ = QFileDialog.getSaveFileName(_parent(), title or "Save", _last_dir(), _filters(file_type))
    _remember(path)
    return path or None


def _message(title, text, kind="info"):
    if kind == "error":
        QMessageBox.critical(_parent(), title, text)
    elif kind == "warning":
        QMessageBox.warning(_parent(), title, text)
    else:
        QMessageBox.information(_parent(), title, text)


def _confirm(title, text):
    r = QMessageBox.question(_parent(), title, text, QMessageBox.Yes | QMessageBox.No)
    return r == QMessageBox.Yes


def _input(title, text, default=""):
    s, ok = QInputDialog.getText(_parent(), title, text, text=default)
    return s if ok else None


def _result_window(table, selection, sd):
    from .result_window import ResultWindow
    ResultWindow(table, selection, sd)


def _cauc_window(target, statistics):
    from .cauc_window import CAUCWindow
    CAUCWindow(target, statistics)


def _cui_domain(paths):
    from .dialogs import CuiDomainChooser
    return CuiDomainChooser(paths, _parent()).get_file()


def _identifier(columns):
    from .dialogs import IdentifierChooser
    return IdentifierChooser(columns, _parent()).choose()


def install_hooks():
    b = common.bridge()
    ui_hooks.install(
        save=lambda file_type=None, title=None: b.call(_save_dialog, file_type, title, wait=True),
        open_=lambda file_type=None, title=None: b.call(_open_dialog, file_type, title, wait=True),
        message=lambda title, text, kind="info": b.call(_message, title, text, kind, wait=True),
        confirm_=lambda title, text: b.call(_confirm, title, text, wait=True),
        input_=lambda title, text, default="": b.call(_input, title, text, default, wait=True))
    ui_hooks.show_result_window = lambda table, selection, sd: b.call(_result_window, table, selection, sd)
    ui_hooks.show_cauc_window = lambda target, stats: b.call(_cauc_window, target, stats)
    ui_hooks.choose_cui_domain = lambda paths: b.call(_cui_domain, paths, wait=True)
    ui_hooks.choose_identifier = lambda columns: b.call(_identifier, columns, wait=True)


def _excepthook(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    try:
        if QApplication.instance() is not None:
            common.show_exception(None, value)
    except Exception:
        pass


def create_app():
    global _app
    if QApplication.instance() is not None:
        _app = QApplication.instance()
        return _app
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    _app = QApplication(sys.argv[:1])
    _app.setApplicationName("SubDisc")
    _app.setOrganizationName("SubDisc")
    _app.setWindowIcon(common.icon())
    theme.apply(_app)
    common.bridge()   # created on the gui thread
    install_hooks()
    sys.excepthook = _excepthook
    return _app


def _has_windows():
    return any(w.isVisible() for w in QApplication.topLevelWidgets())


def main(args=None):
    """SubDisc [file.xml [showWindows [nrThreads]]]

    Without arguments the main window opens (and asks for a data file). With
    an xml autorun file all experiments are run; showWindows 'true' keeps the
    result windows open, otherwise the program ends when done."""
    args = sys.argv[1:] if args is None else list(args)
    if args:
        show_windows = len(args) >= 2 and AttributeType.is_valid_binary_true_value(args[1])
        app = create_app() if show_windows else None
        if xml_auto_run.auto_run_setting(args):
            if app is not None and _has_windows():
                return app.exec()
            return 0

    app = create_app()
    from .mining_window import MiningWindow
    splash = QSplashScreen(common.splash_pixmap())
    splash.show()
    app.processEvents()
    w = MiningWindow()

    def start():
        splash.finish(w)
        w.show()
        w.raise_()
        # java asks for a file before the window is there (except on mac)
        QTimer.singleShot(0, w._open_file)

    QTimer.singleShot(1200, start)
    return app.exec()
