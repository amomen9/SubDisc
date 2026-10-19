"""Small helpers shared by all windows (java GUI class plus a few Qt bits).

Mining runs on a worker thread, Qt widgets may only be touched on the main
thread, so everything that crosses over goes through the Bridge below.
"""

import os
import threading
import traceback

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QColor, QIcon, QPixmap, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (QApplication, QDialog, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget, QComboBox, QProgressBar)

from ..number_format import format_number

RED = QColor(255, 150, 166)
TEXT_FIELD_DEFAULT_SIZE = QSize(86, 22)
WINDOW_DEFAULT_SIZE = QSize(1000, 600)
ROC_WINDOW_DEFAULT_SIZE = QSize(600, 600)
MATRIX_WINDOW_DEFAULT_SIZE = QSize(900, 400)
BUTTON_DEFAULT_SIZE = QSize(160, 25)
BUTTON_MEDIUM_SIZE = QSize(100, 25)

RESOURCES = os.path.join(os.path.dirname(__file__), "resources")

_icon = None


def icon():
    global _icon
    if _icon is None:
        _icon = QIcon(os.path.join(RESOURCES, "icon.png"))
    return _icon


def splash_pixmap():
    return QPixmap(os.path.join(RESOURCES, "cortana.png"))


def with_mnemonic(text, mnemonic):
    """'Close', 'C' -> '&Close', Qt marks the mnemonic with an ampersand"""
    if not mnemonic:
        return text.replace("&", "&&")
    t = text.replace("&", "&&")
    i = t.lower().find(mnemonic.lower())
    if i < 0:
        return t
    return t[:i] + "&" + t[i:]


def build_button(text, mnemonic=None, slot=None, size=BUTTON_DEFAULT_SIZE):
    b = QPushButton(with_mnemonic(text, mnemonic))
    if size is not None:
        b.setMinimumSize(size)
    if slot is not None:
        b.clicked.connect(lambda _=False: slot())
    return b


def build_border(title):
    g = QGroupBox(title)
    return g


def build_text_field(text):
    f = QLineEdit(text)
    f.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    f.setMinimumSize(TEXT_FIELD_DEFAULT_SIZE)
    return f


def build_combo_box(items=(), slot=None):
    c = QComboBox()
    c.setMinimumSize(TEXT_FIELD_DEFAULT_SIZE)
    for s in items:
        c.addItem(s)
    if slot is not None:
        # activated: only user changes, also when the same item is picked
        # again, like a swing ActionEvent
        c.activated.connect(lambda _i: slot())
    return c


def number(x, max_fraction_digits=6):
    """java NumberFormat.getNumberInstance() with a maximum fraction"""
    return format_number(x, max_fraction_digits)


def set_busy(busy):
    if busy:
        QApplication.setOverrideCursor(Qt.WaitCursor)
    else:
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()


def info(parent, title, text):
    QMessageBox.information(parent, title, text)


def warn(parent, title, text):
    QMessageBox.warning(parent, title, text)


def error(parent, title, text):
    QMessageBox.critical(parent, title, text)


# ----------------------------------------------------------------------
# window bookkeeping
# ----------------------------------------------------------------------

# top level windows without a parent are garbage collected in python when
# nobody holds a reference, keep them here until they are closed
_open = set()


def keep(window):
    _open.add(window)
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    window.destroyed.connect(lambda *_: _open.discard(window))
    window.setWindowIcon(icon())
    return window


class Window(QMainWindow):
    """JFrame look alike: a central widget, a button row at the bottom,
    DISPOSE_ON_CLOSE."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        keep(self)
        self.setWindowTitle(title)
        self._central = QWidget()
        self._layout = QVBoxLayout(self._central)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self.setCentralWidget(self._central)

    def content(self):
        return self._layout

    def add_button_row(self, buttons, layout=None):
        row = QHBoxLayout()
        row.addStretch(1)
        for b in buttons:
            row.addWidget(b)
        row.addStretch(1)
        (layout or self._layout).addLayout(row)
        return row

    def show_at(self, x, y, size=None):
        if size is not None:
            self.resize(size)
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()


# ----------------------------------------------------------------------
# calling the gui thread from the mining thread
# ----------------------------------------------------------------------

class Bridge(QObject):
    _call = Signal(object)

    def __init__(self):
        super().__init__()
        self._call.connect(self._run, Qt.QueuedConnection)

    @Slot(object)
    def _run(self, job):
        fn, args, done, box = job
        try:
            box.append(fn(*args))
        except Exception as e:   # pragma: no cover - shown to the user
            box.append(None)
            traceback.print_exc()
            show_exception(None, e)
        finally:
            if done is not None:
                done.set()

    def call(self, fn, *args, wait=False):
        """run fn(*args) on the gui thread, wait=True returns its result"""
        if QThread.currentThread() is self.thread():
            return fn(*args)
        box = []
        done = threading.Event() if wait else None
        self._call.emit((fn, args, done, box))
        if wait:
            done.wait()
            return box[0] if box else None
        return None


_bridge = None


def bridge():
    global _bridge
    if _bridge is None:
        _bridge = Bridge()
    return _bridge


def show_exception(parent, e):
    from .error_window import ErrorWindow
    w = ErrorWindow(e)
    w.move(400, 400)
    w.show()


class BackgroundTask(QObject):
    """Runs fn on a worker thread while a modal dialog is shown.

    The dialog is the java 'Mining in progress...' JOptionPane, with a Stop
    button when on_stop is given. on_done(result) runs on the gui thread."""

    _finished = Signal(object, object)

    def __init__(self, parent, title, text, fn, on_done=None, on_stop=None):
        super().__init__(parent)
        self._fn = fn
        self._on_done = on_done
        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle(title)
        self._dialog.setWindowIcon(icon())
        self._dialog.setModal(True)
        self._dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        lay = QVBoxLayout(self._dialog)
        row = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(self._dialog.style().standardIcon(self._dialog.style().StandardPixmap.SP_MessageBoxInformation).pixmap(32, 32))
        row.addWidget(ic, 0, Qt.AlignTop)
        lbl = QLabel(text)
        row.addWidget(lbl, 1)
        lay.addLayout(row)
        bar = QProgressBar()
        bar.setRange(0, 0)   # busy indicator
        bar.setTextVisible(False)
        bar.setMaximumHeight(8)
        lay.addWidget(bar)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: palette(mid);")
        self.status.setMinimumWidth(380)
        lay.addWidget(self.status)
        self._stop_cb = on_stop
        if on_stop is not None:
            b = QPushButton("Stop")
            b.clicked.connect(self._stop)
            br = QHBoxLayout()
            br.addStretch(1)
            br.addWidget(b)
            br.addStretch(1)
            lay.addLayout(br)
        self._dialog.keyPressEvent = self._key
        self._dialog.reject = lambda: None   # only Stop closes it
        self._finished.connect(self._done, Qt.QueuedConnection)

    def _key(self, e):
        # Escape also stops, like the java dialog
        if e.key() == Qt.Key_Escape and self._stop_cb is not None:
            self._stop()

    def _stop(self):
        if self._stop_cb is not None:
            self._stop_cb()
            self.status.setText("Stopping, the result obtained thus far will be shown...")

    def start(self):
        t = threading.Thread(target=self._work, daemon=True)
        self._dialog.show()
        t.start()

    def _work(self):
        res, exc = None, None
        try:
            res = self._fn()
        except Exception as e:
            traceback.print_exc()
            exc = e
        self._finished.emit(res, exc)

    @Slot(object, object)
    def _done(self, res, exc):
        self._dialog.hide()
        self._dialog.deleteLater()
        if exc is not None:
            show_exception(self.parent(), exc)
        elif self._on_done is not None:
            self._on_done(res)
        self.deleteLater()


def standard_key(seq):
    return QKeySequence(seq)


def screen_center(widget):
    s = QGuiApplication.primaryScreen()
    if s is None:
        return
    g = s.availableGeometry()
    widget.move(g.center() - widget.rect().center())
