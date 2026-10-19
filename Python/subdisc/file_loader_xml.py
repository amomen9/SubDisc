import os

from . import log
from .search_parameters import SearchParameters
from .target_concept import TargetConcept
from .xml_document import XMLDocument


class FileLoaderXML:
    def __init__(self, path, show_windows):
        self._table = None
        self._search_parameters = None
        if path is None or not os.path.exists(path):
            log.log_command_line("FileLoaderXML: can not open File '%s'" % (os.path.abspath(path) if path else None))
            return
        self._load_file(path, show_windows)

    def _load_file(self, path, show_windows):
        from .table import Table
        doc = XMLDocument.parse_xml_file(path)
        if doc is None:
            return
        experiments = list(doc.get_last_child())
        if not experiments:
            return
        settings = list(experiments[0])
        # order matters: Table, SearchParameters, then TargetConcept, so loop backwards
        xml_dir = os.path.dirname(os.path.abspath(path))
        for node in reversed(settings):
            name = node.tag.lower()
            if name == "table":
                self._table = Table.from_xml_node(node, xml_dir, show_windows)
            elif name == "search_parameters":
                self._search_parameters = SearchParameters.from_xml_node(node)
            elif name == "target_concept":
                # NOTE order sensitive, SearchParameters must be set first
                self._search_parameters.set_target_concept(TargetConcept.from_xml_node(node, self._table))

    def get_table(self):
        return self._table

    def get_search_parameters(self):
        return self._search_parameters
