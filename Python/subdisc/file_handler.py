"""Opening and saving files.

The java FileHandler shows the Swing file choosers itself. Here the gui
passes the chosen path (or ui_hooks asks for one), the rest is the same.
"""

import os
from enum import Enum

from . import log
from . import ui_hooks
from .error_log import ErrorLog
from .file_type import FileType
from .timer import Timer


class Action(Enum):
    OPEN_FILE = 0
    OPEN_DATABASE = 1
    SAVE = 2


class FileHandler:
    # directory of the last used file
    last_file_location = "."

    def __init__(self, action=None, path=None, file_type=None, show_windows=True):
        """FileHandler(Action.OPEN_FILE, path) opens a data / xml file,
        FileHandler(Action.SAVE) or FileHandler(file_type=...) asks where to save."""
        self._table = None
        self._search_parameters = None
        self._file = None
        if action == Action.OPEN_FILE:
            if path is None:
                path = ui_hooks.choose_open_file(FileType.ALL_DATA_FILES)
            self._set_file(path)
            self._open_file(show_windows)
        elif action == Action.OPEN_DATABASE:
            self._open_database()
        elif action == Action.SAVE:
            self._save(FileType.ALL_DATA_FILES if file_type is None else file_type, path)
        elif file_type is not None:
            self._save(file_type, path)

    def _set_file(self, path):
        if path:
            self._file = path
            FileHandler.last_file_location = os.path.dirname(os.path.abspath(path))

    @staticmethod
    def populate_from_file(path, table, show_windows):
        """Fill a Table that came from an xml file with the data of path."""
        h = FileHandler()
        if path is None or not os.path.exists(path):
            ErrorLog.log(path, FileNotFoundError(""))
            return h
        if table is None:
            log.log_command_line("FileHandler Constructor: Table is 'null', trying normal loading.")
        h._file = path
        h._table = table
        h._open_file(show_windows)
        return h

    @staticmethod
    def add_cui_domain(table, enrichment_type, domain_file=None):
        """Add a CUI domain to an existing Table."""
        h = FileHandler()
        if table is None:
            log.log_command_line("FileHandler Constructor: parameter can not be 'null'.")
            return h
        from .file_loader_gene_rank import FileLoaderGeneRank
        h._table = FileLoaderGeneRank(table, enrichment_type, domain_file).get_table()
        h._print_loading_info()
        return h

    def _open_file(self, show_windows):
        if self._file is None or not os.path.exists(self._file):
            ErrorLog.log(self._file, FileNotFoundError())
            return
        ft = FileType.get_file_type(self._file)
        timer = Timer()
        if ft == FileType.TXT:
            from .data_loader_txt import DataLoaderTXT
            if self._table is None:
                self._table = DataLoaderTXT(self._file).get_table()
            else:
                # load from XML, see Table.from_xml_node()
                DataLoaderTXT(self._file, self._table)
        elif ft == FileType.ARFF:
            from .file_loader_arff import FileLoaderARFF
            if self._table is None:
                self._table = FileLoaderARFF(self._file).get_table()
            else:
                FileLoaderARFF(self._file, self._table)
        elif ft == FileType.XML:
            from .file_loader_xml import FileLoaderXML
            loader = FileLoaderXML(self._file, show_windows)
            self._table = loader.get_table()
            self._search_parameters = loader.get_search_parameters()
            return   # does not print the loading info / time
        else:
            log.log_command_line("FileHandler: unknown FileType for File '%s'." % os.path.basename(self._file))
            return
        log.log_command_line("loading time '%s': %s" % (self._file, timer.get_elapsed_time_string()))
        if self._table is not None:
            self._print_loading_info()

    def _open_database(self):
        pass

    def _save(self, file_type, path=None):
        if path is None:
            path = ui_hooks.choose_save_file(file_type)
        self._set_file(path)
        return self._file

    def _print_loading_info(self):
        self._table.update()
        log.log_command_line("Table '%s' has %d columns and %d rows." % (
            self._table.get_name(), self._table.get_nr_columns(), self._table.get_nr_rows()))

    def get_file(self):
        """chosen file, None when nothing was approved"""
        return self._file

    def get_table(self):
        return self._table

    def get_search_parameters(self):
        return self._search_parameters
