"""ARFF loader (weka format). TODO from java: rewrite, file loading is a mess."""

import os
import re

from . import log
from .attribute_type import AttributeType
from .column import Column, DEFAULT_INIT_SIZE
from .java_compat import parse_float, jtrim, float_to_string
from .table import Table

MISSING_NUMERIC = float('nan')


class _Keyword:
    def __init__(self, keyword):
        self.text = re.compile(r"^\s*" + keyword + r"\s*", re.IGNORECASE)

    # FIXME MM (java) - returns true if KEYWORD occurs anywhere in String
    def at_start_of_line(self, s):
        return self.text.search(s) is not None


COMMENT = _Keyword("%")
RELATION = _Keyword("@relation")
ATTRIBUTE = _Keyword("@attribute")
DATA = _Keyword("@data")
END = _Keyword("@end")


def remove_outer_quotes(s):
    """first unescaped quote pair"""
    if s.startswith("'"):
        i = 1
        n = len(s)
        while i < n:
            if s[i] == '\\':   # jump beyond escaped char
                i += 1
            elif s[i] == "'":
                return s[1:i]
            i += 1
        log.log_command_line("FileLoaderARFF: could not find a closing ' for String:\n\t" + s + "\nReturning whole String.")
    return s


class FileLoaderARFF:
    def __init__(self, path, table=None):
        self._table = None
        self._check_data_with_xml_table = False
        self._nr_data_rows = 0
        self._nr_bad_rows = 0
        if path is None or not os.path.exists(path):
            log.log_command_line("FileLoaderARFF: can not open File '%s'" % (os.path.abspath(path) if path else None))
            return
        if table is None:
            self._load_file(path)
        else:
            self._check_data_with_xml_table = True
            self._table = table
            self._load_file(path)

    def _load_file(self, path):
        relation_found = False
        data_found = False
        attribute_index = 0
        name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline=None) as f:
                for line in f:
                    if line.endswith("\n"):
                        line = line[:-1]
                    if COMMENT.at_start_of_line(line) or line == "":
                        continue
                    elif END.at_start_of_line(line):
                        missing = ""
                        if not relation_found:
                            missing += "\nNo '@relation' found."
                        if attribute_index == 0:
                            missing += "\nNo '@attribute' found."
                        if not data_found:
                            missing += "\nNo '@data' found."
                        if missing:
                            log.log_command_line("FileLoaderARFF: premature '@end' declaration in File '" + name + "'." + missing)
                        break
                    elif RELATION.at_start_of_line(line):
                        if not relation_found:
                            relation_found = True
                            if self._check_data_with_xml_table:
                                continue
                            parts = re.split(r"\s", line, maxsplit=1)
                            self._table = Table(path, remove_outer_quotes(parts[1] if len(parts) > 1 else ""))
                        else:
                            log.log_command_line("FileLoaderARFF: multiple '@relation' declarations in File '%s', using: '%s'."
                                                 % (name, self._table.get_name()))
                    elif ATTRIBUTE.at_start_of_line(line):
                        if not relation_found:
                            if self._check_data_with_xml_table:
                                continue
                            break
                        m = ATTRIBUTE.text.search(line)
                        if m:
                            if self._check_data_with_xml_table:
                                c = self._parse_attribute(line[m.end():], attribute_index)
                                if c.get_name() == self._table.get_column(attribute_index).get_name():
                                    attribute_index += 1
                                    continue
                                break
                            self._table.get_columns().append(self._parse_attribute(line[m.end():], attribute_index))
                            attribute_index += 1
                    elif DATA.at_start_of_line(line):
                        if not relation_found:
                            break
                        elif attribute_index == 0:
                            break
                        elif data_found:
                            continue
                        else:
                            data_found = True
                    elif data_found:
                        self._load_data(jtrim(line), self._nr_data_rows)
                        self._nr_data_rows += 1
                    if self._nr_data_rows > 0 and self._nr_data_rows % 10000 == 0:
                        log.log_command_line("loadFile: " + str(self._nr_data_rows) + " lines read")
            if self._nr_bad_rows > 0:
                log.log_command_line("FileLoaderARFF: " + str(self._nr_bad_rows) + " offending lines encounterd.")
            # exact array lengths, see DataLoaderTXT
            if self._table is not None:
                for c in self._table.get_columns():
                    c.close()
        except OSError as e:
            log.log_command_line(str(e))

    def _parse_attribute(self, line, index):
        """only NUMERIC / NOMINAL / BINARY for now, STRING / DATE become nominal"""
        if line.startswith("'"):
            name = remove_outer_quotes(line)
        else:
            name = re.split(r"\s", line, maxsplit=1)[0]
        # FIXME MM (java) quick hack, remove the name, then the first space
        rest = jtrim(line).replace(name, "")
        parts = re.split(r"\s", rest, maxsplit=1)
        rest = jtrim(parts[1]) if len(parts) > 1 else ""
        return Column(name, None, self._declared_type(name, rest), index, DEFAULT_INIT_SIZE)

    def _load_data(self, line, line_nr):
        original = line
        bad = False
        for c in self._table.get_columns():
            # XXX MM - ugly hack for now
            line = jtrim(line)
            if line.startswith("'"):
                cell = remove_outer_quotes(line)
                # use offset, there might be ',' inside the quotes
                idx = line.find(',', len(cell) + 2)
                if idx >= 0:
                    line = line[idx + 1:]
                else:
                    line = line[len(cell) + 2:]
            else:
                idx = line.find(",")
                cell = jtrim(line[:idx]) if idx >= 0 else jtrim(line)
                if idx >= 0:
                    line = line[idx + 1:]
                else:
                    line = line[len(cell):]
            if cell == "?":
                c.add_missing()
            else:
                bad |= not self._add_value_to_column(c, cell, line_nr)
        if bad:
            if self._nr_bad_rows == 0:
                log.log_command_line("FileLoaderARFF: more lines with erroneous values for numeric columns will not be reported per line.")
            self._nr_bad_rows += 1
        if line != "":
            if not COMMENT.at_start_of_line(line):
                log.log_command_line("FileLoaderARFF: many arguments at line " + str(line_nr) + ":\n\t" + line + "\n\t" + original)

    def _add_value_to_column(self, column, cell, line_nr):
        t = column.get_type()
        if t == AttributeType.NOMINAL:
            column.add_nominal(cell)
        elif t == AttributeType.NUMERIC:
            try:
                column.add_float(parse_float(cell))
            except ValueError:
                column.add_float(MISSING_NUMERIC)
                if self._nr_bad_rows == 0:
                    cause = "missing value" if cell == "" else "unparsable value '" + cell + "'"
                    log.log_command_line("FileLoaderARFF: line %d, %s in numeric column '%s', inserting %s."
                                         % (line_nr, cause, column.get_name(), "NaN"))
                return False
        elif t == AttributeType.BINARY:
            # TODO (java) fails for autorun data with 2 values set to BINARY by hand
            if not AttributeType.is_valid_binary_value(cell):
                raise ValueError("FileLoaderARFF: invalid BINARY value: " + cell)
            column.add_binary(AttributeType.is_valid_binary_true_value(cell))
        else:
            raise AssertionError(t)
        return True

    def _declared_type(self, attribute_name, s):
        low = s.lower()
        if low.startswith("real") or low.startswith("integer") or low.startswith("numeric"):
            return AttributeType.NUMERIC
        if low.startswith("{"):
            rest = s[1:]
            classes = []
            while not rest.startswith("}"):
                offset = 0
                if rest.startswith("'"):
                    nominal_class = remove_outer_quotes(rest)
                    offset = 3
                else:
                    if "," in rest:
                        nominal_class = re.split(r",\s*", rest, maxsplit=1)[0]
                        offset = 1
                    else:
                        nominal_class = rest.split("}", 1)[0]
                        classes.append(nominal_class)
                        break
                classes.append(nominal_class)
                rest = re.sub(r",\s*", "", rest[len(nominal_class) + offset:], count=1)
                if rest == "":
                    break
            if len(classes) == 2:
                a, b = classes
                if ((AttributeType.is_valid_binary_true_value(a) and AttributeType.is_valid_binary_false_value(b))
                        or (AttributeType.is_valid_binary_true_value(b) and AttributeType.is_valid_binary_false_value(a))):
                    return AttributeType.BINARY
            # TODO (java) present attributeType change dialog to user
            return AttributeType.NOMINAL
        # TODO (java) parse dates
        return AttributeType.NOMINAL

    def get_table(self):
        return self._table
