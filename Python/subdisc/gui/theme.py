"""Look and feel. Java uses FlatLaf IntelliJ (flat, light, rounded corners
of 5px, chevron arrows); this is a Fusion based copy of it, plus a dark
variant (FlatLaf Darcula like) that can be picked in the View menu."""

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

LIGHT = "light"
DARK = "dark"

_COLORS = {
    LIGHT: dict(window="#f2f2f2", base="#ffffff", alt="#f7f8fa", text="#000000",
                button="#ffffff", border="#c4c4c4", accent="#2675bf",
                accent_soft="#d5e4f2", disabled="#8c8c8c", group="#555555",
                header="#f2f2f2", tooltip="#f7f7f7"),
    DARK: dict(window="#3c3f41", base="#45494a", alt="#414445", text="#bbbbbb",
               button="#4c5052", border="#5e6060", accent="#4a88c7",
               accent_soft="#2f65ca", disabled="#777777", group="#a9a9a9",
               header="#3c3f41", tooltip="#4b4d4d"),
}

_QSS = """
QWidget {{ font-family: "Segoe UI", "Helvetica Neue", "DejaVu Sans", Arial, sans-serif; font-size: 9pt; }}
QMainWindow, QDialog {{ background: {window}; }}
QGroupBox {{
    border: 1px solid {border}; border-radius: 5px;
    margin-top: 12px; padding: 8px 6px 6px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: {group}; font-weight: bold;
}}
QPushButton {{
    background: {button}; border: 1px solid {border}; border-radius: 5px;
    padding: 4px 12px; min-height: 18px;
}}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {accent_soft}; }}
QPushButton:default, QPushButton:focus {{ border: 1px solid {accent}; }}
QPushButton:disabled {{ color: {disabled}; border-color: {border}; }}
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background: {base}; border: 1px solid {border}; border-radius: 5px;
    padding: 2px 6px; selection-background-color: {accent};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {accent}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {disabled}; background: {window}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {base}; border: 1px solid {border};
    selection-background-color: {accent}; selection-color: white;
}}
QTableView, QListWidget, QListView, QTreeView {{
    background: {base}; alternate-background-color: {alt};
    border: 1px solid {border}; gridline-color: {border};
    selection-background-color: {accent}; selection-color: white;
}}
QHeaderView::section {{
    background: {header}; border: none;
    border-right: 1px solid {border}; border-bottom: 1px solid {border};
    padding: 3px 6px;
}}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 4px; }}
QTabBar::tab {{ padding: 5px 12px; border: none; }}
QTabBar::tab:selected {{ border-bottom: 2px solid {accent}; }}
QMenuBar {{ background: {window}; }}
QMenu {{ background: {base}; border: 1px solid {border}; }}
QMenu::item:selected {{ background: {accent}; color: white; }}
QStatusBar {{ color: {group}; }}
QToolTip {{ background: {tooltip}; color: {text}; border: 1px solid {border}; }}
QProgressBar {{ border: 1px solid {border}; border-radius: 3px; background: {base}; }}
QProgressBar::chunk {{ background: {accent}; }}
"""


def current():
    try:
        v = QSettings("SubDisc", "SubDisc").value("theme", LIGHT)
        return v if v in _COLORS else LIGHT
    except Exception:
        return LIGHT


def apply(app=None, name=None):
    app = app or QApplication.instance()
    name = name or current()
    c = _COLORS[name]
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(c["window"]))
    p.setColor(QPalette.WindowText, QColor(c["text"]))
    p.setColor(QPalette.Base, QColor(c["base"]))
    p.setColor(QPalette.AlternateBase, QColor(c["alt"]))
    p.setColor(QPalette.Text, QColor(c["text"]))
    p.setColor(QPalette.Button, QColor(c["button"]))
    p.setColor(QPalette.ButtonText, QColor(c["text"]))
    p.setColor(QPalette.Highlight, QColor(c["accent"]))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipBase, QColor(c["tooltip"]))
    p.setColor(QPalette.ToolTipText, QColor(c["text"]))
    p.setColor(QPalette.Mid, QColor(c["disabled"]))
    p.setColor(QPalette.PlaceholderText, QColor(c["disabled"]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(c["disabled"]))
    app.setPalette(p)
    app.setStyleSheet(_QSS.format(**c))
    try:
        QSettings("SubDisc", "SubDisc").setValue("theme", name)
    except Exception:
        pass


def is_dark():
    return current() == DARK
