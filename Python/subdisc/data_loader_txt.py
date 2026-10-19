"""Default loader for delimited text files (.txt .csv .tsv).

The column type is guessed from the first data line and changed when a
later line does not fit (binary -> numeric/nominal, numeric -> nominal),
the same way the java DataLoaderTXT does it.
"""

import os
import re

from . import log
from .attribute_type import AttributeType
from .column import Column
from .java_compat import parse_float, jtrim
from .table import Table

DELIMITERS = ('\t', ',', ';')

_EMPTY = re.compile(r"[ \t\n\x0b\f\r]*")


def _read_lines(path):
    # BufferedReader.readLine(): \n, \r and \r\n end a line
    with open(path, "r", encoding="utf-8", errors="replace", newline=None) as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]
            yield line


def scanner_tokens(line, delim):
    """Tokens of new Scanner(line).useDelimiter(delim): split, but one
    leading and one trailing delimiter do not give an empty token."""
    if line == "":
        return []
    parts = line.split(delim)
    if line.startswith(delim):
        parts = parts[1:]
    if line.endswith(delim) and parts:
        parts = parts[:-1]
    return parts


def is_empty_string(s):
    return _EMPTY.fullmatch(s) is not None


def remove_quotes(s):
    """strips quotes at the start and end, works best after trim()"""
    if s == "" or (s[0] != '"' and s[0] != "'"):
        return s
    n = len(s)
    if ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")) and n > 2:
        s = s[1:-1]
    return s


def opens_quotes(s):
    """only opens a quote, without closing it?"""
    if s == "":
        return False
    a, b = s[0], s[-1]
    if a != "'" and a != '"':
        return False
    if a == '"' and b == '"':
        return False
    if a == "'" and b == "'":
        return False
    return True


def closes_quotes(s):
    """only closes a quote, without opening it?"""
    if s == "":
        return False
    a, b = s[0], s[-1]
    if a == '"' and len(s) > 1:
        return False
    if a == "'" and len(s) > 1:
        return False
    return b == "'" or b == '"'


def _is_float(s):
    try:
        parse_float(s)
        return True
    except ValueError:
        return False


class DataLoaderTXT:
    def __init__(self, path, table=None, xml_mode=False):
        """With a table (xml loading) the columns are known already and only
        the data is read into them."""
        self._table = None
        self._delimiter = 0
        self._nr_lines = 0

        warning = None
        if path is None:
            warning = "file can not be null"
        elif not os.path.exists(path):
            warning = os.path.abspath(path) + ", file does not exist"
        elif not os.access(path, os.R_OK):
            warning = os.path.abspath(path) + ", file not readable"
        if warning is not None:
            self._message("<init>", warning)
            return
        if xml_mode or table is not None:
            self._table = table
            if self._table is None:
                self._message("<init>", "Table is null, attempting regular file-load.")
        self._load_file(path)

    def _message(self, method, msg):
        log.log_command_line("%s.%s(): %s" % (self.__class__.__name__, method, msg))

    def _load_file(self, path):
        # analyse establishes the number of data lines and the delimiter
        if not self._analyse(path):
            return
        delim = self.get_delimiter_string()
        lines = _read_lines(path)
        header = None
        line = None
        line_nr = 0
        print_trigger = 1000
        print_update = 1000

        # skip to the header, not empty
        for line in lines:
            line_nr += 1
            if line != "":
                header = line
                break

        original_types = None
        if self._table is not None:
            original_types = self._check_xml_table(header, path)
            if original_types is None:
                return
        else:
            # first data line decides the column types
            for line in lines:
                line_nr += 1
                if line != "":
                    self._create_table(path, header, line)
                    break
            if self._table is None:
                return

        columns = self._table.get_columns()
        nr_columns = len(columns)
        binaries = [False] * nr_columns
        floats = [False] * nr_columns
        true_values = [None] * nr_columns
        false_values = [None] * nr_columns

        # true / false values of the columns that look binary
        tokens = scanner_tokens(line if line is not None else "", delim)
        for c, tok in enumerate(tokens[:nr_columns]):
            if columns[c].get_type() == AttributeType.BINARY:
                s = remove_quotes(tok)
                if AttributeType.is_valid_binary_value(s):
                    if AttributeType.is_valid_binary_true_value(s):
                        true_values[c] = s
                    else:
                        false_values[c] = s

        for i, c in enumerate(columns):
            if c.get_type() == AttributeType.BINARY:
                binaries[i] = True
            elif c.get_type() == AttributeType.NUMERIC:
                floats[i] = True

        self._message("loadFile", "loading data")
        d0 = self.get_delimiter()
        # code ignores AttributeType.ORDINAL
        for line in lines:
            line_nr += 1
            if line == "":
                continue
            if line[0] == d0:          # is the first field missing?
                line = " " + line
            if line[-1] == d0:         # is the last field missing?
                line = line + " "
            toks = scanner_tokens(line, delim)

            col = -1
            t = 0
            nt = len(toks)
            while t < nt and col < nr_columns - 1:
                col += 1
                s = toks[t]
                t += 1
                if opens_quotes(s):
                    # the delimiter came before the quote was closed
                    while t < nt:
                        nxt = toks[t]
                        t += 1
                        s = s + delim + nxt
                        if closes_quotes(nxt):
                            break
                s = remove_quotes(jtrim(s))
                column = columns[col]

                if binaries[col]:
                    # still binary? (that may change while reading)
                    if is_empty_string(s):
                        column.add_missing()
                        continue
                    if AttributeType.is_valid_binary_value(s):
                        v = AttributeType.is_valid_binary_true_value(s)
                        column.add_binary(v)
                        if v:
                            true_values[col] = s
                        else:
                            false_values[col] = s
                        continue
                    # neither missing nor binary, so not binary
                    binaries[col] = False
                    try:
                        f = parse_float(s)
                        floats[col] = True
                        column.set_type(AttributeType.NUMERIC)
                        column.add_float(f)
                        log.log_command_line(column.get_name() + " was binary, is numeric (line " + str(line_nr) + ")")
                    except ValueError:
                        column.to_nominal_type(true_values[col], false_values[col])
                        column.add_nominal(s)
                        log.log_command_line(column.get_name() + " was binary, is nominal (line " + str(line_nr) + ")")
                elif floats[col]:
                    if is_empty_string(s):
                        column.add_missing()
                    else:
                        try:
                            column.add_float(parse_float(s))
                        except ValueError:
                            # guess it's a nominal then
                            floats[col] = False
                            column.set_type(AttributeType.NOMINAL)
                            column.add_nominal(s)
                            log.log_command_line(column.get_name() + " was float, is nominal (line " + str(line_nr) + ")")
                else:
                    if is_empty_string(s):
                        column.add_missing()
                    else:
                        column.add_nominal(s)

            if col != nr_columns - 1 or t < nt:
                found = col + 1 + (nt - t)
                self._message("loadFile", "incorrect number of fields on line " + str(line_nr) + ". "
                              + str(nr_columns) + " expected, " + str(found) + " found.")
                # java leaves the short columns as they are (or crashes on
                # too many fields), here missing values keep the rows aligned
                for k in range(col + 1, nr_columns):
                    columns[k].add_missing()

            if line_nr == print_trigger:
                self._message("loadFile", str(line_nr) + " lines read")
                print_trigger += print_update
                if print_trigger == print_update * 10 and print_update != 10000:
                    print_update *= 10
                    self._message("loadFile", "number of lines for update interval changed to: " + str(print_update))

        for c in columns:
            log.log_command_line("Column " + c.get_name() + " (" + str(c.get_type()) + ")")
            # java leaves the arrays at their grown capacity (its loops only
            # look at itsSize), numpy masks need the exact length
            c.close()

        if original_types is not None:
            self._evaluate_xml_loading(original_types, path)

    def _analyse(self, path):
        self._message("analyse", "analysing " + os.path.abspath(path))
        try:
            lines = _read_lines(path)
            header = None
            for header in lines:
                if header != "":
                    break
            nr_data = 0
            for line in lines:
                if line != "":
                    nr_data += 1
                    self._establish_delimiter(header, line)
                    break
            for line in lines:
                if line != "":
                    nr_data += 1
            self._message("analyse", str(nr_data) + " lines of data found")
            self._nr_lines = nr_data
            return True
        except OSError:
            self._message("analyse", "IOException caused by file: " + os.path.abspath(path))
            return False

    def _establish_delimiter(self, first, second):
        counts = [len(first.split(d)) for d in DELIMITERS]
        nr_options = sum(1 for c in counts if c > 1)
        msg = ""
        if nr_options == 0:
            msg = "unable to determine delimiter, using '" + DELIMITERS[0] + "'"
        elif nr_options == 1:
            for i, c in enumerate(counts):
                if c > 1:
                    self._delimiter = i
                    msg = "successfully established delimiter, using '" + DELIMITERS[i] + "'"
        else:
            for i, c in enumerate(counts):
                if c > 1:
                    # just pick the first one that fits the second line
                    if c == len(second.split(DELIMITERS[i])):
                        self._delimiter = i
                        msg = "unsure about delimiter, using '" + DELIMITERS[i] + "'"
                        break
            # sic, the java code always ends with this message
            msg = "unable to determine delimiter, using '" + DELIMITERS[0] + "'"
        self._message("establishDelimiter", msg)

    def _check_xml_table(self, header, path):
        """header names must match the xml declared columns, returns the
        xml declared types"""
        headers = header.split(self.get_delimiter_string())
        n = len(headers)
        bad = False
        if n != len(self._table.get_columns()):
            self._message("checkXMLTable", "ERROR\nNumber of Columns declared in XML: %d\nNumber of Columns retrieved from File %s: %d"
                          % (len(self._table.get_columns()), os.path.basename(path), n))
            bad = True
        for i in range(min(n, len(self._table.get_columns()))):
            if headers[i] != self._table.get_column(i).get_name():
                self._message("checkXMLTable", "ERROR on index %d\nColumn '%s' from XML does not match Column '%s' from File '%s'"
                              % (i + 1, self._table.get_column(i).get_name(), jtrim(headers[i]), os.path.basename(path)))
                bad = True
                break
        if bad:
            return None
        return [self._table.get_column(i).get_type() for i in range(n)]

    def _create_table(self, path, header, data_line):
        """columns from the header names, types from the first data line"""
        self._message("createTable", "creating Table")
        delim = self.get_delimiter_string()
        headers = [remove_quotes(jtrim(h)) for h in header.split(delim)]
        data = data_line.split(delim)
        self._table = Table(path, None, self._nr_lines, len(headers))
        columns = self._table.get_columns()
        for i, h in enumerate(headers):
            if i < len(data):
                s = remove_quotes(data[i])
            else:
                # java throws here, treat it as a missing value
                s = ""
            # binary (or empty)
            if AttributeType.is_valid_binary_value(s) or is_empty_string(s):
                columns.append(Column(h, None, AttributeType.BINARY, i, self._nr_lines))
                if is_empty_string(s):
                    columns[i].add_missing()
                else:
                    columns[i].add_binary(AttributeType.is_valid_binary_true_value(s))
                continue
            # numeric
            try:
                f = parse_float(s)
                columns.append(Column(h, None, AttributeType.NUMERIC, i, self._nr_lines))
                columns[i].add_float(f)
                continue
            except ValueError:
                pass
            # ordinal: no use case yet
            columns.append(Column(h, None, AttributeType.NOMINAL, i, self._nr_lines))
            columns[i].add_nominal(s)

    def _evaluate_xml_loading(self, original_types, path):
        for i, t in enumerate(original_types):
            if self._table.get_column(i).get_type() != t:
                self._message("evaluateXMLLoading", "WARNING Column '%s'\n\tXML declared AttributeType: '%s'\n\tAttributeType after parsing File '%s': '%s'"
                              % (self._table.get_column(i).get_name(), t, os.path.abspath(path), self._table.get_column(i).get_type()))

    def get_table(self):
        # TODO (java) still returns a table when no data was loaded
        return self._table

    def get_delimiter(self):
        return DELIMITERS[self._delimiter]

    def get_delimiter_string(self):
        return DELIMITERS[self._delimiter]
