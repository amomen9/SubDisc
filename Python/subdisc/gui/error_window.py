"""java ErrorWindow: the message of an exception in a text area."""

from PySide6.QtWidgets import QPlainTextEdit

from . import common


class ErrorWindow(common.Window):
    def __init__(self, exception):
        super().__init__("Error")
        t = QPlainTextEdit()
        t.setReadOnly(True)
        t.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        msg = str(exception) if exception is not None else ""
        t.setPlainText(msg + "\n\n")
        self.content().addWidget(t)
        b = common.build_button("Close", "C", self.close, size=None)
        b.setMinimumWidth(142)
        self.add_button_row([b])
        self.resize(360, 380)
