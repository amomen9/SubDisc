import os
from enum import Enum

from . import log


class FileType(Enum):
    TXT = (0, "Text Files", (".txt", ".text", ".csv", ".tsv"))
    ARFF = (1, "ARFF Files", (".arff",))
    XML = (2, "XML Files", (".xml",))
    PLT = (3, "GnuPlot Files", (".gp", ".plt"))
    ALL_DATA_FILES = (4, "Data Files", None)

    @property
    def DESCRIPTION(self):
        return self.value[1]

    def get_extensions(self):
        if self is FileType.ALL_DATA_FILES:
            return list(FileType.TXT.value[2]) + list(FileType.ARFF.value[2]) + list(FileType.XML.value[2])
        return list(self.value[2])

    def qt_filter(self):
        """Filter string for a QFileDialog, e.g. 'Text Files (*.txt *.csv)'."""
        return "%s (%s)" % (self.DESCRIPTION, " ".join("*" + e for e in self.get_extensions()))

    @staticmethod
    def remove_extension(path):
        """File name without everything from the last '.' on."""
        if path is None or not os.path.isfile(path):
            return ""
        name = os.path.basename(path)
        i = name.rfind('.')
        return name if i < 0 else name[:i]

    @staticmethod
    def get_file_type(path):
        from .error_log import ErrorLog
        if path is None or not os.path.exists(path):
            ErrorLog.log(path, FileNotFoundError())
        name = os.path.basename(path).lower()
        i = name.rfind('.')
        ext = name[i:] if i >= 0 else name
        for ft in FileType:
            if ext in ft.get_extensions():
                return ft
        log.log_command_line("FileType.getFileType(): The extension '%s' of File '%s', can not be resolved to a known FileType." % (ext, name))
        return None
