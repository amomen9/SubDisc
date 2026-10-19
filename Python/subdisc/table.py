import os

import numpy as np

from . import log
from . import ui_hooks
from .attribute_type import AttributeType
from .column import Column
from .file_type import FileType
from .java_compat import JavaRandom, float_to_string
from .target_type import TargetType


class Table:
    # NOTE when adding members, update the constructors AND select()

    def __init__(self, source=None, name=None, nr_rows=0, nr_columns=0):
        """source is a file path; when name is None it comes from the file
        name (like the TXT loader constructor)."""
        self._source = os.path.basename(source) if source else source
        if name is None and source is not None:
            name = FileType.remove_extension(source)
        self._name = name
        self._nr_rows = nr_rows
        self._nr_columns = nr_columns
        self._columns = []
        self._random_number = JavaRandom(10)
        self._domains = None          # TODO (java) better never None
        self._domain_indices = None   # allows much faster removal

    @staticmethod
    def from_xml_node(node, xml_dir, show_windows):
        """Table from a <table> element, then the data is loaded from the
        source file next to the xml file."""
        from .file_handler import FileHandler
        t = Table()
        for child in node:
            tag = child.tag.lower()
            if tag == "table_name":
                t._name = child.text or ""
            elif tag == "source":
                t._source = child.text or ""
            elif tag == "column":
                t._columns.append(Column.from_xml_node(child))
        # all columns are known, check them by loading the real data
        FileHandler.populate_from_file(os.path.join(xml_dir, t._source), t, show_windows)
        return t

    @staticmethod
    def from_result_set(cursor, name):
        """PSR database, any DB-API cursor after execute()."""
        t = Table()
        t._name = name
        t._source = "PSR"
        desc = cursor.description
        t._nr_columns = len(desc)
        rows = cursor.fetchall()
        for i, d in enumerate(desc):
            col_name = d[0]
            type_code = str(d[1]).upper() if d[1] is not None else ""
            sample = next((r[i] for r in rows if r[i] is not None), None)
            nominal = ("CHAR" in type_code or "DATE" in type_code or isinstance(sample, str))
            t._columns.append(Column(col_name, col_name, AttributeType.NOMINAL if nominal else AttributeType.NUMERIC, i, 0))
        nulls = False
        for r in rows:
            for i, c in enumerate(t._columns):
                v = r[i]
                if v is None:
                    log.log_command_line("NULL value encountered in column " + c.get_name())
                    nulls = True
                if c.get_type() == AttributeType.NOMINAL:
                    c.add_nominal("" if v is None else str(v))
                else:
                    c.add_float(0.0 if v is None else float(v))
            t._nr_rows += 1
        if nulls:
            ui_hooks.show_message("NULL values encountered", "The were NULL values in one or more columns.")
        return t

    def get_name(self):
        return self._name

    def get_source(self):
        return self._source

    # NOTE nr_columns is not tied to len(columns)
    def get_nr_rows(self):
        return self._nr_rows

    def get_nr_columns(self):
        return self._nr_columns

    def get_columns(self):
        return self._columns

    def get_column(self, key):
        """by index (int) or by name (str), None when the name is unknown"""
        if isinstance(key, str):
            for c in self._columns:
                if c.get_name() == key:
                    return c
            return None
        return self._columns[key]

    def get_index(self, name):
        for c in self._columns:
            if c.get_name() == name:
                return c.get_index()
        return -1

    # ------------------------------------------------------------------
    # CUI domains (bioinformatics module)
    # ------------------------------------------------------------------

    def add_domain(self, domain_name):
        """relies on the caller for update()"""
        if self._domains is None:
            self._domains = []
            self._domain_indices = []
        if domain_name in self._domains:
            log.log_command_line("A domain with the name '%s' is already present." % domain_name)
            return False
        self._domains.append(domain_name)
        # a rank column is inserted when there is none, the offset keeps it
        # from being removed with the first domain
        if len(self._domain_indices) == 0:
            offset = 1
            for c in self._columns:
                if c.get_name().lower() == "rank":
                    offset -= 1
                    break
            self._domain_indices.append(len(self._columns) + offset)
        else:
            self._domain_indices.append(len(self._columns))
        return True

    def remove_domain(self, domain_index):
        if self._domains is None or domain_index < 0 or domain_index > len(self._domains) - 1:
            log.log_command_line("Domain '%s' not found." % domain_index)
            return
        self._domains.pop(domain_index)
        start = self._domain_indices.pop(domain_index)
        if domain_index == len(self._domain_indices):
            end = len(self._columns)
        else:
            end = self._domain_indices[domain_index]
        del self._columns[start:end]
        deleted = end - start
        for i in range(domain_index, len(self._domain_indices)):
            self._domain_indices[i] -= deleted
        self._nr_columns = len(self._columns)
        if start < self._nr_columns:
            for i in range(start, self._nr_columns):
                self._columns[i].set_index(i)
        if not self._domains:
            self._domains = None
            self._domain_indices = None

    def get_domain_list(self):
        return [] if self._domains is None else list(self._domains)

    def update(self):
        """Sets the number of rows and columns, closes the columns."""
        self._nr_rows = self._columns[0].size() if self._columns else 0
        self._nr_columns = len(self._columns)   # needed for MiningWindow
        for c in self._columns:
            c.close()
            c.get_cardinality()

    def get_type_counts(self):
        """[[total, enabled]] for NOMINAL, NUMERIC, ORDINAL, BINARY"""
        counts = [[0, 0] for _ in range(4)]
        order = {AttributeType.NOMINAL: 0, AttributeType.NUMERIC: 1, AttributeType.ORDINAL: 2, AttributeType.BINARY: 3}
        for c in self._columns:
            k = order[c.get_type()]
            counts[k][0] += 1
            if c.get_is_enabled():
                counts[k][1] += 1
        return counts

    def evaluate(self, condition_list):
        """members for a ConditionList (empty list gives an empty BitSet)"""
        if condition_list.size() == 0:
            return np.zeros(0, dtype=bool)
        b = np.ones(self._nr_rows, dtype=bool)
        for i in range(condition_list.size()):
            c = condition_list.get(i)
            b = c.get_column().evaluate(b, c)
        return b

    def get_random_bit_set(self, nr_bits_to_set):
        """BitSet of table size with nr_bits_to_set bits set (selection sampling)."""
        sample = np.zeros(self._nr_rows, dtype=bool)
        m = 0
        t = 0
        rnd = self._random_number
        for i in range(self._nr_rows):
            th1 = float(nr_bits_to_set - m)
            th2 = float(self._nr_rows - t)
            if th2 * rnd.next_double() < th1:
                sample[i] = True
                m += 1
                t += 1
                if m >= nr_bits_to_set:
                    break
            else:
                t += 1
        return sample

    def select(self, mask):
        """New Table with only the rows in mask (not a true deep copy)."""
        result = Table(None, self._name, int(np.count_nonzero(mask)), self._nr_columns)
        result._source = self._source
        for c in self._columns:
            result._columns.append(c.select(mask))
        result._random_number = self._random_number
        result._domains = self._domains
        result._domain_indices = self._domain_indices
        result.update()
        return result

    def swap_randomize_target(self, tc):
        """NOTE destroys the target columns of tc, back them up first."""
        from .validation import Validation
        t = tc.get_target_type()
        if not Validation.is_valid_random_qualities_target_type(t):
            raise ValueError("Table.swapRandomizeTarget(): !Validation.isValidRandomQualitiesTargetType(%s)" % t)
        if t in (TargetType.SINGLE_NOMINAL, TargetType.SINGLE_NUMERIC, TargetType.LABEL_RANKING):
            targets = [tc.get_primary_target()]
        elif t in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY):
            targets = [tc.get_primary_target(), tc.get_secondary_target()]
        elif t == TargetType.MULTI_LABEL:
            targets = tc.get_multi_targets()
        else:
            raise AssertionError("Table.swapRandomizeTarget(): unimplemented TargetType '%s'" % t)
        n = self.get_nr_rows()
        perm = list(range(n))
        rnd = self._random_number
        for i in range(n - 1):
            j = i + rnd.next_int(n - i)
            perm[i], perm[j] = perm[j], perm[i]
        for c in targets:
            log.log_command_line("permuting \"" + c.get_name() + "\"")
            c.permute(perm)

    def print(self):
        log.log_command_line("Types ===========================================")
        for c in self._columns:
            c.print()
        log.log_command_line("Table ===========================================")
        for i in range(self._columns[0].size()):
            log.log_command_line("Row %d: %s" % (i + 1, ", ".join(c.get_string(i) for c in self._columns)))
        log.log_command_line("=================================================")

    def _cell(self, k, i):
        c = self._columns[k]
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            return c.get_nominal(i)
        if t in (AttributeType.NUMERIC, AttributeType.ORDINAL):
            return float_to_string(c.get_float(i))
        if t == AttributeType.BINARY:
            return "1" if c.get_binary(i) else "0"
        log.log_command_line("Unknown AttributeType: " + str(t))
        return ""

    def to_file(self, members=None, path=None):
        """Write (a selection of) this Table as comma separated text. Without
        a path the gui asks for one (members given) like the java version."""
        if path is None:
            path = ui_hooks.choose_save_file(FileType.ALL_DATA_FILES)
            if path is None:
                return
        rows = range(self._nr_rows) if members is None else [int(i) for i in np.flatnonzero(members) if i < self._nr_rows]
        try:
            with open(path, "w", encoding="utf-8", newline="") as w:
                w.write(",".join(self._columns[i].get_name() for i in range(self._nr_columns)))
                w.write("\n")
                for i in rows:
                    w.write(",".join(self._cell(k, i) for k in range(self._nr_columns)))
                    w.write("\n")
                w.write("\n")
        except OSError:
            log.log_command_line("Error while writing: " + str(path))

    def add_node_to(self, parent):
        from .xml_node import add_node_to
        node = add_node_to(parent, "table")
        add_node_to(node, "table_name", self._name)
        add_node_to(node, "source", self._source)
        for c in self._columns:
            c.add_node_to(node)
