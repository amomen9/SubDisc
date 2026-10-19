"""Creating, parsing and saving the xml files (autorun, mrml, patternset)."""

import traceback
import xml.etree.ElementTree as ET
from enum import Enum

from . import log


class XMLType(Enum):
    AUTORUN = "autorun"
    MRML = "mrml"
    PATTERNSET = "patternset"

    @property
    def qualified_name(self):
        return self.value

    @property
    def system_id(self):
        return self.value + ".dtd"


class XMLDocument:
    """Root element plus the doctype it should be written with."""

    def __init__(self, root, system_id):
        self.root = root
        self.system_id = system_id

    def get_last_child(self):
        # the document element, like getLastChild() on a java Document
        return self.root

    @staticmethod
    def build_document(xml_type):
        return XMLDocument(ET.Element(xml_type.qualified_name), xml_type.system_id)

    @staticmethod
    def parse_xml_file(path):
        """Parses the file and numbers the top level nodes with an id attribute."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            _strip_whitespace(root)
            for i, child in enumerate(list(root)):
                child.set("id", str(i))
            sys_id = root.tag + ".dtd"
            return XMLDocument(root, sys_id)
        except Exception as e:
            log.log_command_line(str(e))
            traceback.print_exc()
            return None

    @staticmethod
    def save_document(doc, path):
        try:
            _indent(doc.root)
            body = ET.tostring(doc.root, encoding="unicode", short_empty_elements=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
                f.write('<!DOCTYPE %s SYSTEM "%s">\n' % (doc.root.tag, doc.system_id))
                f.write(body)
                f.write("\n")
        except Exception:
            traceback.print_exc()


def _strip_whitespace(elem):
    # same as setIgnoringElementContentWhitespace(true)
    if elem.text is not None and len(elem) > 0 and elem.text.strip() == "":
        elem.text = None
    for child in elem:
        if child.tail is not None and child.tail.strip() == "":
            child.tail = None
        _strip_whitespace(child)


def _indent(elem, level=0, space="  "):
    pad = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + space
        for child in elem:
            _indent(child, level + 1, space)
            if not child.tail or not child.tail.strip():
                child.tail = pad + space
        last = elem[-1]
        last.tail = pad
    if level == 0:
        elem.tail = None
