"""Autorun xml files: save experiments, run them from the command line and
write the result sets as tab separated text."""

import os
import sys
import time
from datetime import datetime
from enum import Enum

from . import log
from . import condition_list_builder as clb
from .attribute_type import AttributeType
from .file_type import FileType
from .java_compat import jstr, double_to_string
from .result_table_model import get_column_name, COLUMN_COUNT
from .xml_document import XMLDocument, XMLType

# leave False: write conditions in canonical order (invocation invariant)
IS_MULTI_THREAD_TEST = False
RESULT_SET_DELIMITER = "\t"


class AutoRun(Enum):
    CREATE = 0
    ADD = 1


class XMLAutoRun:
    def __init__(self, search_parameters, table, file_option_or_path):
        """XMLAutoRun(sp, table, AutoRun.CREATE|ADD) asks for a file (gui),
        XMLAutoRun(sp, table, path) writes a new file."""
        if search_parameters is None or table is None or file_option_or_path is None:
            raise ValueError("XMLAutoRun<init>: arguments can not be null")
        self._document = None
        if isinstance(file_option_or_path, AutoRun):
            from .file_handler import FileHandler
            path = FileHandler(file_type=FileType.XML).get_file()
            if path is None:
                return
            self._build_document(path, search_parameters, table, file_option_or_path)
        else:
            self._build_document(file_option_or_path, search_parameters, table, AutoRun.CREATE)

    def _build_document(self, path, sp, table, option):
        if option == AutoRun.CREATE or not os.path.exists(path):
            self._document = XMLDocument.build_document(XMLType.AUTORUN)
        else:
            self._document = XMLDocument.parse_xml_file(path)
        if self._document is None:
            return
        root = self._document.get_last_child()
        root.append(self._build_experiment_element(sp, table))
        # only after inserting the new node
        root.set("nr_experiments", str(len(list(root))))
        XMLDocument.save_document(self._document, path)

    def _build_experiment_element(self, sp, table):
        import xml.etree.ElementTree as ET
        root = self._document.get_last_child()
        # reset all experiment ids, the file may have been edited by hand
        children = list(root)
        for i, c in enumerate(children):
            c.set("id", str(i))
        node = ET.Element("experiment")
        node.set("id", str(len(children)))
        sp.get_target_concept().add_node_to(node)
        sp.add_node_to(node)
        table.add_node_to(node)
        return node


def auto_run_setting(args):
    """Runs every experiment of an autorun file. True if args were valid."""
    if len(args) == 0:
        return False
    path = None
    show_windows = False
    nr_threads = -2147483648
    if not args[0].endswith(".xml") or len(args) > 3:
        show_help()
    else:
        path = args[0]
    if len(args) >= 2:
        show_windows = AttributeType.is_valid_binary_true_value(args[1])
    if len(args) == 3:
        try:
            from .java_compat import parse_int
            nr_threads = parse_int(args[2])
        except ValueError:
            show_help()
    run_all_from_file(path, show_windows, nr_threads)
    return True


def run_all_from_file(path, show_windows, nr_threads):
    doc = XMLDocument.parse_xml_file(path)
    if doc is None:
        return []
    results = []
    for exp in list(doc.get_last_child()):
        results.append(run_subgroup_discovery(exp, path, show_windows, nr_threads))
    return results


def run_subgroup_discovery(experiment_node, path, show_windows, nr_threads):
    from .process import run_subgroup_discovery as run
    from .search_parameters import SearchParameters
    from .table import Table
    from .target_concept import TargetConcept
    settings = list(experiment_node)
    parent = os.path.dirname(path) or "."
    table = Table.from_xml_node(settings[2], parent, show_windows)
    table.update()
    sp = SearchParameters.from_xml_node(settings[1])
    # the command line may overrule the number of threads in the file
    if nr_threads == -2147483648:
        nr_threads = sp.get_nr_threads()
    sp.set_target_concept(TargetConcept.from_xml_node(settings[0], table))
    begin = time.time()
    sd = run(table, 0, None, sp, show_windows, nr_threads, None)
    stamp = datetime.fromtimestamp(begin).strftime("%Y%m%d%H%M%S") + "%03d" % int((begin % 1) * 1000)
    out = os.path.abspath(path).replace(".xml", "_" + stamp + ".txt")
    save(sd.get_result(), out, sp.get_target_type())
    return out


def save(subgroup_set, file_name, target_type):
    if subgroup_set is None or file_name is None:
        return
    d = RESULT_SET_DELIMITER
    try:
        with open(file_name, "w", encoding="utf-8", newline="") as w:
            w.write(get_column_name(0, target_type))
            for i in range(1, COLUMN_COUNT):
                w.write(d)
                w.write(get_column_name(i, target_type))
            w.write("\n")
            for s in subgroup_set:
                w.write(str(s.get_id()))
                w.write(d)
                w.write(str(s.get_depth()))
                w.write(d)
                w.write(str(s.get_coverage()))
                w.write(d)
                w.write(double_to_string(s.get_measure_value()))
                w.write(d)
                w.write(double_to_string(s.get_secondary_statistic()))
                w.write(d)
                w.write(double_to_string(s.get_tertiary_statistic()))
                w.write(d)
                w.write(double_to_string(s.get_p_value()))
                w.write(d)
                if IS_MULTI_THREAD_TEST:
                    w.write(clb.to_canonical_order_string(s.get_conditions()))
                else:
                    w.write(str(s.get_conditions()))
                w.write("\n")
    except OSError:
        log.log_command_line("Error on file: " + str(file_name))


def show_help():
    log.log_command_line("")
    log.log_command_line("Usage: python -m subdisc /path/to/file.xml [showWindows] [nrThreads]")
    log.log_command_line("")
    log.log_command_line("filepath can be relative")
    log.log_command_line("filename must end with '.xml'")
    log.log_command_line("")
    log.log_command_line("optional showWindows:")
    log.log_command_line("'true' to show result window")
    log.log_command_line("'false' to suppress all GUI elements (default)")
    log.log_command_line("")
    log.log_command_line("optional nrThreads:")
    log.log_command_line("positive integer indicating the number of threads to use")
    log.log_command_line("default is " + str(os.cpu_count()))
    log.log_command_line("(the python port runs the search in one thread anyway)")
    log.log_command_line("")
    sys.exit(0)
