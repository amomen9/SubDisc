"""One column of data (a Table attribute).

Storage follows the java class: NOMINAL columns keep an int code per row
plus the list of distinct values, NUMERIC columns a float32 array, BINARY
columns a bool array (the BitSet). Missing values ('?' in the file) are
flagged in a separate bool array.

BitSets from the java code are numpy bool arrays here, always as long as
the column (so a subgroup's members are a mask over all rows).
"""

import math
import re

import numpy as np

from . import log
from .attribute_type import AttributeType, DEFAULT_BINARY_TRUE_STRING, DEFAULT_BINARY_FALSE_STRING
from .column_statistics import Statistics
from .java_compat import (float_to_string, parse_float, float_compare, java_sort_floats, to_f32, fdiv,
                          parse_int, parse_boolean)
from . import function
from .operators import Operator

DEFAULT_INIT_SIZE = 2048
MAP_DEFAULT_INIT_SIZE = DEFAULT_INIT_SIZE // 32

# all NUMERIC Conditions set a sort index, Column.evaluate() relies on it;
# UNINITIALISED_SORT_INDEX is MIN_VALUE so outside the valid range
MASK_ON = np.int32(-2147483648)   # 0x80000000
MASK_OFF = np.int32(0x7fffffff)

_TRUE_FLOAT = re.compile(r"\+?0*1(\.0+)?")
_TRUE_INTEGER = re.compile(r"[-+]?\d+(\.0+)?")

NONE = 0
PRIMARY = 1
SECONDARY = 2
TERTIARY = 3
FIRST_TARGET_STATUS = NONE
LAST_TARGET_STATUS = TERTIARY

_TARGET_TEXT = {NONE: " none", PRIMARY: " primary", SECONDARY: " secondary", TERTIARY: " tertiary"}

USE_NEW_BINNING = False   # see get_unique_split_points(), leave False


def _order_keys(a):
    """Integer keys that sort float32 values like Float.compare() does."""
    a = np.asarray(a, dtype=np.float32)
    b = a.view(np.int32).astype(np.int64)
    # canonical NaN like floatToIntBits
    b = np.where(np.isnan(a), np.int64(0x7fc00000), b)
    return np.where(b < 0, b ^ np.int64(0x7FFFFFFF), b)


def _seq_sum_f32(vals):
    """sum in float32, one value after the other, like a java float loop."""
    if vals.size == 0:
        return 0.0
    with np.errstate(all='ignore'):
        return float(np.cumsum(vals, dtype=np.float32)[-1])


def _grow(arr, needed):
    if needed <= arr.shape[0]:
        return arr
    new_cap = max(needed, arr.shape[0] * 2, 16)
    out = np.zeros(new_cap, dtype=arr.dtype)
    out[:arr.shape[0]] = arr
    return out


class DomainMapNumeric:
    """values at [>= size] are invalid, -1 in counts at [size] marks the end"""
    __slots__ = ("itsSize", "itsCountsSum", "itsDomain", "itsCounts")

    def __init__(self, size, counts_sum, domain, counts):
        self.itsSize = size
        self.itsCountsSum = counts_sum
        self.itsDomain = domain
        self.itsCounts = counts


class ValueCount:
    __slots__ = ("itsCounts",)

    def __init__(self, counts):
        self.itsCounts = counts


class ValueCountTP:
    __slots__ = ("itsCounts", "itsTruePositives", "itsMissingCount", "itsMissingPositiveCount")

    def __init__(self, counts, tps, missing_count, missing_positive_count):
        self.itsCounts = counts
        self.itsTruePositives = tps
        self.itsMissingCount = missing_count
        self.itsMissingPositiveCount = missing_positive_count


class ValueCountSum:
    __slots__ = ("itsCounts", "itsSums")

    def __init__(self, counts, sums):
        self.itsCounts = counts
        self.itsSums = sums


class ValueCountSumSquaresSum:
    __slots__ = ("itsCounts", "itsSums", "itsSquaresSums")

    def __init__(self, counts, sums, squares_sums):
        self.itsCounts = counts
        self.itsSums = sums
        self.itsSquaresSums = squares_sums


class Column:
    # NOTE when adding members, update __init__, copy() and select()

    def __init__(self, name, short, attribute_type, index, nr_rows):
        self._type = None
        self._name = None
        self._short = short
        self._index = 0
        self._floatz = None
        self._nominalz = None
        self._binaries = None
        self._distinct_values = None
        self._distinct_values_map = None
        self._missing = np.zeros(0, dtype=bool)
        self._missing_value_is_unique = True
        self._size = 0
        self._cardinality = 0
        self._min = float('inf')
        self._max = float('-inf')
        self._is_enabled = True
        self._sorted_floats = None
        self._sort_index = None
        self._target_status = NONE
        self._distinct_values_u = None

        if index is None:
            # used by from_xml_node()
            return
        if not self._is_valid_index(index):
            return
        self._index = index
        self._check_and_set_name(name)
        self._check_and_set_type(attribute_type)
        self._setup_column(DEFAULT_INIT_SIZE if nr_rows <= 0 else nr_rows)
        self._target_status = SECONDARY if self._type == AttributeType.NUMERIC else NONE

    @staticmethod
    def from_xml_node(node):
        """Column from a <column> element of an autorun / xml file."""
        from .xml_node import text_of
        c = Column(None, None, None, None, 0)
        for child in node:
            tag = child.tag.lower()
            text = text_of(child)
            if tag == "type":
                c._type = AttributeType.from_string(text)
            elif tag == "name":
                c._name = text
            elif tag == "short":
                c._short = text
            elif tag == "index":
                c._index = parse_int(text)
            elif tag == "enabled":
                c._is_enabled = parse_boolean(text)
            else:
                raise AssertionError("unknown Node: " + child.tag)
        c._setup_column(DEFAULT_INIT_SIZE)
        return c

    def _is_valid_index(self, index):
        ok = index >= 0
        if not ok:
            log.log_command_line("Column<init>: index can not be < 0. No index set.")
        return ok

    def _check_and_set_name(self, name):
        if not name:
            import time
            self._name = str(time.perf_counter_ns())
            log.log_command_line("Column<init>: name can not be 'null' or empty. Using: " + self._name)
        else:
            self._name = name

    def _check_and_set_type(self, attribute_type):
        if attribute_type is None:
            self._type = AttributeType.get_default()
            log.log_command_line("Column<init>: type can not be 'null'. Using: " + self._type.name)
        else:
            self._type = attribute_type

    def set_index(self, index):
        """called after removing (domain) columns from Table"""
        if self._is_valid_index(index):
            self._index = index

    def _setup_column(self, nr_rows):
        t = self._type
        if t == AttributeType.NOMINAL:
            self._nominalz = np.zeros(nr_rows, dtype=np.int32)
            self._distinct_values = []
            self._distinct_values_map = {}
        elif t == AttributeType.NUMERIC:
            self._floatz = np.zeros(nr_rows, dtype=np.float32)
        elif t == AttributeType.BINARY:
            self._binaries = np.zeros(nr_rows, dtype=bool)
        elif t == AttributeType.ORDINAL:
            raise AssertionError(t)
        else:
            self._log_type_error("Column<init>")
            raise AssertionError(t)
        self._missing = np.zeros(nr_rows, dtype=bool)

    def add_node_to(self, parent):
        """Adds the <column> element (name, type etc.), not the data."""
        from .xml_node import add_node_to
        node = add_node_to(parent, "column")
        add_node_to(node, "type", str(self._type))
        add_node_to(node, "name", self._name)
        add_node_to(node, "short", "" if self._short is None else self._short)
        add_node_to(node, "index", self._index)
        add_node_to(node, "enabled", self._is_enabled)

    def copy(self):
        c = Column(self._name, self._short, self._type, self._index, self._size)
        c._floatz = self._floatz
        c._nominalz = self._nominalz
        c._binaries = self._binaries
        c._distinct_values = self._distinct_values
        c._distinct_values_map = self._distinct_values_map
        c._missing = self._missing
        c._missing_value_is_unique = self._missing_value_is_unique
        c._size = self._size
        c._cardinality = self._cardinality
        c._min = self._min
        c._max = self._max
        c._is_enabled = self._is_enabled
        c._target_status = self._target_status
        return c

    def select(self, mask):
        """Copy with only the rows set in mask. Not a deep copy; missing,
        cardinality and min/max are not copied (should be re-assessed)."""
        mask = np.asarray(mask, dtype=bool)
        idx = np.flatnonzero(mask[:self._size]) if mask.shape[0] >= self._size else np.flatnonzero(mask)
        idx = idx[idx < self._size]
        n = int(np.count_nonzero(mask))
        c = Column(self._name, self._short, self._type, self._index, n)
        c._distinct_values = self._distinct_values
        c._distinct_values_map = self._distinct_values_map
        c._size = n
        c._is_enabled = self._is_enabled
        c._target_status = self._target_status
        t = self._type
        if t == AttributeType.NOMINAL:
            out = np.zeros(n, dtype=np.int32)
            out[:idx.size] = self._nominalz[idx]
            c._nominalz = out
        elif t == AttributeType.NUMERIC:
            out = np.zeros(n, dtype=np.float32)
            out[:idx.size] = self._floatz[idx]
            c._floatz = out
        elif t == AttributeType.BINARY:
            out = np.zeros(max(self._size, n), dtype=bool)
            out[:idx.size] = self._binaries[idx]
            c._binaries = out
        else:
            self._log_type_error("Column.select()")
            raise AssertionError(t)
        c._missing = np.zeros(n, dtype=bool)
        return c

    # ------------------------------------------------------------------
    # adding data
    # ------------------------------------------------------------------

    def add_missing(self):
        self.set_missing(self._size)
        t = self._type
        if t == AttributeType.NOMINAL:
            self.add_nominal("?")
        elif t == AttributeType.NUMERIC:
            self.add_float(float('nan'))
        elif t == AttributeType.BINARY:
            self.add_binary(False)
        else:
            self._log_type_error("Column.addMissing()")
            raise AssertionError(t)

    def add(self, value):
        """Dispatch on the python type, like the java overloads."""
        if isinstance(value, (bool, np.bool_)):
            self.add_binary(bool(value))
        elif isinstance(value, str):
            self.add_nominal(value)
        else:
            self.add_float(value)

    def add_nominal(self, value):
        if value is None:
            raise TypeError("null")
        if self._size == self._nominalz.shape[0]:
            self._nominalz = _grow(self._nominalz, self._size * 2 if self._size else 16)
        i = self._distinct_values_map.get(value)
        if i is None:
            i = len(self._distinct_values)
            self._distinct_values.append(value)
            self._distinct_values_map[value] = i
        self._nominalz[self._size] = i
        self._size += 1

    def add_float(self, value):
        if self._size == self._floatz.shape[0]:
            self._floatz = _grow(self._floatz, self._size * 2 if self._size else 16)
        self._floatz[self._size] = value
        self._size += 1

    def add_binary(self, value):
        if self._size >= self._binaries.shape[0]:
            self._binaries = _grow(self._binaries, self._size + 1)
        if value:
            self._binaries[self._size] = True
        self._size += 1

    def set_data(self, values):
        values = list(values) if not isinstance(values, np.ndarray) else values
        t = self._type
        if t == AttributeType.NOMINAL:
            for s in values:
                self.add_nominal(s)
        elif t == AttributeType.NUMERIC:
            self._floatz = np.array(values, dtype=np.float32)
            self._size = self._floatz.shape[0]
        elif t == AttributeType.BINARY:
            b = np.array(values, dtype=bool)
            self._binaries = _grow(self._binaries, b.shape[0])
            self._binaries[:b.shape[0]] = b
            self._size = b.shape[0]
        self.close()

    def close(self):
        """Always call this after the last element is added."""
        if self._nominalz is not None and self._nominalz.shape[0] > self._size:
            self._nominalz = self._nominalz[:self._size].copy()
        if self._floatz is not None and self._floatz.shape[0] > self._size:
            self._floatz = self._floatz[:self._size].copy()
        if self._binaries is not None and self._binaries.shape[0] < self._size:
            self._binaries = _grow(self._binaries, self._size)
        if self._missing.shape[0] < self._size:
            self._missing = _grow(self._missing, self._size)
        if self._missing.shape[0] > self._size:
            self._missing = self._missing[:self._size].copy()
        if self._binaries is not None and self._binaries.shape[0] > self._size:
            self._binaries = self._binaries[:self._size].copy()

    def set(self, index, value):
        """package private, for FileLoaderGeneRank only"""
        if not self._is_out_of_bounds(index):
            self._floatz[index] = value

    # ------------------------------------------------------------------
    # getters
    # ------------------------------------------------------------------

    def size(self):
        return self._size

    def get_name(self):
        return self._name

    def get_short(self):
        return self._short if self.has_short() else ""

    def has_short(self):
        return self._short is not None

    def get_name_and_short(self):
        return self._name + ((" (" + self.get_short() + ")") if self.has_short() else "")

    def get_type(self):
        return self._type

    def get_index(self):
        return self._index

    def get_nominal(self, index):
        return "" if self._is_out_of_bounds(index) else self._distinct_values[int(self._nominalz[index])]

    def get_float(self, index):
        return float('nan') if self._is_out_of_bounds(index) else float(self._floatz[index])

    def get_binary(self, index):
        return False if self._is_out_of_bounds(index) else bool(self._binaries[index])

    def get_string(self, index):
        t = self._type
        if t == AttributeType.NOMINAL:
            return self.get_nominal(index)
        if t == AttributeType.NUMERIC:
            return float_to_string(self.get_float(index))
        if t == AttributeType.BINARY:
            return "1" if self.get_binary(index) else "0"
        if t == AttributeType.ORDINAL:
            raise AssertionError(t)
        self._log_type_error("Column.getString()")
        raise AssertionError(t)

    def get_binaries(self):
        """copy of the bool data (BINARY only)"""
        return self._binaries[:self._size].copy()

    def get_floats(self):
        """copy of the float data (NUMERIC only)"""
        return self._floatz.copy()

    # fast read only access for the mining code, do not modify
    def floats_view(self):
        return self._floatz

    def binaries_view(self):
        return self._binaries[:self._size]

    def nominal_codes_view(self):
        return self._nominalz

    def distinct_values(self):
        return self._distinct_values

    def distinct_values_map(self):
        return self._distinct_values_map

    def missing_view(self):
        return self._missing[:self._size]

    def _is_out_of_bounds(self, index):
        out = index < 0 or index >= self._size
        if out:
            log.log_command_line("indexOutOfBounds: " + str(index))
        return out

    def get_min(self):
        self._update_min_max()
        return self._min

    def get_max(self):
        self._update_min_max()
        return self._max

    def _update_min_max(self):
        # not computed yet?
        if self._max == float('-inf'):
            vals = self._floatz[:self._size]
            finite = vals[~np.isnan(vals)]
            if finite.size:
                mx = float(finite.max())
                mn = float(finite.min())
                if mx > self._max:
                    self._max = mx
                if mn < self._min:
                    self._min = mn

    def permute(self, permutation):
        """Destructive swap-randomisation, make a copy() first if needed."""
        perm = np.asarray(permutation, dtype=np.int64)
        t = self._type
        if t == AttributeType.NOMINAL:
            self._nominalz = self._nominalz[perm].copy()
        elif t == AttributeType.NUMERIC:
            self._floatz = self._floatz[perm].copy()
        elif t == AttributeType.BINARY:
            self._binaries = self._binaries[perm].copy()
        elif t == AttributeType.ORDINAL:
            raise AssertionError(t)
        else:
            self._log_type_error("Column.permute()")
            raise AssertionError(t)

    def print(self):
        log.log_command_line(str(self._index) + ":" + self.get_name_and_short() + " " + str(self._type))
        t = self._type
        if t == AttributeType.NOMINAL:
            log.log_command_line("[" + ", ".join(str(int(v)) for v in self._nominalz) + "]")
        elif t == AttributeType.NUMERIC:
            log.log_command_line("[" + ", ".join(float_to_string(v) for v in self._floatz) + "]")
        elif t == AttributeType.BINARY:
            log.log_command_line("{" + ", ".join(str(int(i)) for i in np.flatnonzero(self._binaries[:self._size])) + "}")
        else:
            raise AssertionError(t)

    # ------------------------------------------------------------------
    # changing the type
    # ------------------------------------------------------------------

    def set_type(self, attribute_type):
        """True when the change worked."""
        if self._type == attribute_type:
            return True
        if attribute_type == AttributeType.NOMINAL:
            return self._to_nominal_type()
        if attribute_type == AttributeType.NUMERIC:
            return self._to_numeric_type(attribute_type)
        if attribute_type == AttributeType.BINARY:
            return self._to_binary_type()
        if attribute_type == AttributeType.ORDINAL:
            raise AssertionError(self._type)
        self._log_type_error("Column.setType()")
        raise AssertionError(attribute_type)

    def _to_nominal_type(self):
        """Any type can become NOMINAL."""
        assert self._type != AttributeType.NOMINAL
        n = self._size
        self._distinct_values = []
        self._distinct_values_map = {}
        self._nominalz = np.zeros(n, dtype=np.int32)
        self._size = 0
        if self._type == AttributeType.NUMERIC:
            # NOTE in java an 'all values are integers' test sits here, but
            # it compares against itsSize after resetting it to 0, so it only
            # ever fires when no value is an integer, and then it crashes on
            # split(".")[0]. Every run that does not crash prints the value
            # with Float.toString(), so that is what happens here.
            floats = self._floatz
            for i in range(n):
                self.add_nominal(float_to_string(floats[i]))
            self._floatz = None
        elif self._type == AttributeType.BINARY:
            bins = self._binaries
            for i in range(n):
                self.add_nominal(DEFAULT_BINARY_TRUE_STRING if bins[i] else DEFAULT_BINARY_FALSE_STRING)
            self._binaries = None
        else:
            self._log_type_error("Column.toNominalType()")
            raise AssertionError(self._type)
        self._type = AttributeType.NOMINAL
        return True

    def to_nominal_type(self, a_true, a_false):
        """During loading: a column that looked BINARY turns out to be nominal.
        Needs at least one of the two values that were read as binary."""
        assert self._type == AttributeType.BINARY
        n = self._size
        self._distinct_values = []
        self._distinct_values_map = {}
        self._nominalz = np.zeros(max(n, 16), dtype=np.int32)
        self._size = 0
        bins = self._binaries
        if a_true is None and a_false is None:
            # just missing values so far
            for i in range(n):
                self.add_nominal("?")
        else:
            for i in range(n):
                # java would add a null here when one of the two is unknown,
                # which throws, so this case never got through either
                v = a_true if (i < bins.shape[0] and bins[i]) else a_false
                self.add_nominal(v if v is not None else "?")
        self._binaries = None
        self._type = AttributeType.NOMINAL
        return True

    def _to_numeric_type(self, new_type):
        """NOMINAL only works when every value parses as a float."""
        t = self._type
        if t == AttributeType.NOMINAL:
            n = self._size
            out = np.zeros(n, dtype=np.float32)
            # parse every distinct value once, same outcome as per row
            parsed = {}
            for code in np.unique(self._nominalz[:n]) if n else []:
                try:
                    parsed[int(code)] = parse_float(self._distinct_values[int(code)])
                except ValueError:
                    return False
            if n:
                lut = np.zeros(len(self._distinct_values), dtype=np.float32)
                for k, v in parsed.items():
                    lut[k] = v
                out[:] = lut[self._nominalz[:n]]
            self._floatz = out
            self._distinct_values = None
            self._distinct_values_map = None
            self._nominalz = None
            if not self._missing[:self._size].any():
                self._missing_value_is_unique = not bool(np.isnan(out).any())
        elif t == AttributeType.NUMERIC:
            pass
        elif t == AttributeType.BINARY:
            out = np.zeros(self._size, dtype=np.float32)
            out[self._binaries[:self._size]] = 1.0
            self._floatz = out
            self._binaries = None
        elif t == AttributeType.ORDINAL:
            raise AssertionError(t)
        else:
            self._log_type_error("Column.toNumericType()")
            raise AssertionError(t)
        self._type = new_type
        return True

    def _to_binary_type(self):
        """At most two distinct values. The first value of the column is the
        'true' case, unless the values themselves read as true/false."""
        assert self._type != AttributeType.BINARY
        card = self.get_cardinality()
        if card > 2 or card < 0:
            return False
        n = self._size
        t = self._type
        if t == AttributeType.NOMINAL:
            bins = np.zeros(n, dtype=bool)
            if n > 0:
                codes = self._nominalz[:n]
                v = int(codes[0])
                value = self._distinct_values[v]
                if AttributeType.is_valid_binary_true_value(value):
                    bins[codes == v] = True
                elif AttributeType.is_valid_binary_false_value(value):
                    bins[codes != v] = True
                elif value == "?":
                    # sets all non-missing to 'true'
                    miss = self._missing[:n]
                    clear = np.flatnonzero(~miss)
                    if clear.size and AttributeType.is_valid_binary_true_value(self._distinct_values[int(codes[clear[0]])]):
                        bins[clear] = True
                else:
                    bins[codes == v] = True
            self._binaries = bins
            self._distinct_values = None
            self._distinct_values_map = None
            self._nominalz = None
        elif t == AttributeType.NUMERIC:
            # all values other than zero become true (NaN too)
            f = self._floatz[:n]
            self._binaries = (f != 0.0) | np.isnan(f)
            self._floatz = None
        else:
            self._log_type_error("Column.toBinaryType()")
            raise AssertionError(t)
        # all missing to false
        self._binaries[self._missing[:n]] = False
        self._update_cardinality()
        self._type = AttributeType.BINARY
        return True

    def _log_type_error(self, method):
        log.log_command_line("Error in %s: Column '%s' has AttributeType '%s'." % (method, self.get_name(), self._type))

    # ------------------------------------------------------------------

    def get_is_enabled(self):
        return self._is_enabled

    def set_is_enabled(self, setting):
        self._is_enabled = bool(setting)

    def get_has_missing_values(self):
        return bool(self._missing.any())

    def get_missing(self, index=None):
        """Copy of the missing flags, or one flag when an index is given."""
        if index is None:
            return self._missing[:self._size].copy()
        return bool(self._missing[index]) if index < self._missing.shape[0] else False

    def set_missing(self, index):
        if index >= self._missing.shape[0]:
            self._missing = _grow(self._missing, index + 1)
        self._missing[index] = True

    def _is_valid_value(self, attribute_type, value):
        if attribute_type == AttributeType.NOMINAL:
            return True
        if attribute_type == AttributeType.NUMERIC:
            try:
                parse_float(value)
                return True
            except ValueError:
                return False
        if attribute_type == AttributeType.BINARY:
            return AttributeType.is_valid_binary_true_value(value) or AttributeType.is_valid_binary_false_value(value)
        raise AssertionError(attribute_type)

    def get_cardinality(self):
        """Number of distinct values, best called after all data is loaded."""
        if self._size == 0 or self._size == 1:
            return self._size
        if self._cardinality != 0:
            return self._cardinality
        t = self._type
        if t == AttributeType.NOMINAL:
            self._missing_value_is_unique = True
            self._cardinality = len(self._distinct_values)
            return self._cardinality
        if t == AttributeType.NUMERIC:
            f = self._floatz[:self._size]
            if (np.isnan(f) & ~self._missing[:self._size]).any():
                self._missing_value_is_unique = False
            self._cardinality = function.get_cardinality(self._floatz)
            return self._cardinality
        if t == AttributeType.BINARY:
            b = self._binaries[:self._size]
            if (~b & ~self._missing[:self._size]).any():
                self._missing_value_is_unique = False
            self._cardinality = self._get_binaries_cardinality()
            return self._cardinality
        self._log_type_error("Column.getNrDistinct()")
        raise AssertionError(t)

    def _update_cardinality(self):
        if self._cardinality == 0:
            self.get_cardinality()
        else:
            b = self._binaries[:self._size]
            if (~b & ~self._missing[:self._size]).any():
                self._missing_value_is_unique = False
        self._cardinality = self._get_binaries_cardinality()

    def _get_binaries_cardinality(self):
        """0 if empty, 1 if all 0s or all 1s, 2 if both"""
        if self._size == 0:
            return 0
        b = self._binaries[:self._size]
        has_true = bool(b.any())
        if not has_true:
            return 1
        return 2 if not bool(b.all()) else 1

    def make_no_target(self):
        if self._type == AttributeType.NUMERIC:
            self._target_status = NONE

    def make_primary_target(self):
        if self._type == AttributeType.NUMERIC:
            self._target_status = PRIMARY

    def make_secondary_target(self):
        if self._type == AttributeType.NUMERIC:
            self._target_status = SECONDARY

    def make_tertiary_target(self):
        if self._type == AttributeType.NUMERIC:
            self._target_status = TERTIARY

    def get_target_status(self):
        return self._target_status

    def display_target_status(self):
        return Column.get_target_text(self._target_status)

    @staticmethod
    def get_target_text(status):
        if status not in _TARGET_TEXT:
            raise AssertionError("unknown TargetStatus: " + str(status))
        return _TARGET_TEXT[status]

    def set_target_status(self, status):
        if self._type != AttributeType.NUMERIC:
            return
        if isinstance(status, str):
            for k, v in _TARGET_TEXT.items():
                if v == status:
                    self._target_status = k
                    return
            self._target_status = 2147483647   # should be impossible
        else:
            self._target_status = int(status)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(self, members, condition):
        """Rows in members for which the condition holds (members is not changed)."""
        if members is None:
            raise ValueError("BitSet can not be null")
        # XXX evaluation logic is deeply flawed, this hack is needed
        if condition.get_column() is not self:
            raise ValueError("%s does not apply to %s" % (condition, self.get_name()))
        op = condition.get_operator()
        n = self._size
        members = members[:n] if members.shape[0] > n else members
        t = self._type
        if t == AttributeType.NOMINAL:
            if op == Operator.ELEMENT_OF:
                vs = condition.get_nominal_value_set()
                lut = np.array([vs.contains(v) for v in self._distinct_values], dtype=bool)
                res = members & lut[self._nominalz[:n]] if lut.size else np.zeros(n, dtype=bool)
            elif op == Operator.EQUALS:
                # NOTE this removes X = '?' from the results, intentionally
                v = self._distinct_values_map[condition.get_nominal_value()]
                res = members & (self._nominalz[:n] == v) & ~self._missing[:n]
            else:
                raise AssertionError(str(t) + " incompatible with " + str(op))
        elif t == AttributeType.NUMERIC:
            if op == Operator.BETWEEN:
                iv = condition.get_numeric_interval()
                res = members & iv.between_mask(self._floatz[:n]) & ~self._missing[:n]
            elif op in (Operator.EQUALS, Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL):
                res = members & self._numeric_compare(op, condition.get_numeric_value(), condition.get_sort_index()) & ~self._missing[:n]
            else:
                raise AssertionError(str(t) + " incompatible with " + str(op))
        elif t == AttributeType.BINARY:
            assert op == Operator.EQUALS
            res = self.evaluate_binary(members, condition.get_binary_value())
        elif t == AttributeType.ORDINAL:
            raise AssertionError(t)
        else:
            raise AssertionError(t)
        return res

    def _numeric_compare(self, op, value, sort_index):
        from .condition import UNINITIALISED_SORT_INDEX
        n = self._size
        if self._sort_index is not None and sort_index != UNINITIALISED_SORT_INDEX:
            si = self._sort_index & MASK_OFF
            if op == Operator.EQUALS:
                return si == sort_index
            if op == Operator.LESS_THAN_OR_EQUAL:
                return si <= sort_index
            return si >= sort_index
        f = self._floatz[:n]
        v = np.float32(value)
        if op == Operator.EQUALS:
            return f == v
        if op == Operator.LESS_THAN_OR_EQUAL:
            return f <= v
        return f >= v

    def evaluate_binary(self, members, value):
        """BINARY evaluation straight on the members (see the java notes)."""
        assert self._type == AttributeType.BINARY
        n = members.shape[0]
        b = self._binaries[:n] if self._binaries.shape[0] >= n else _grow(self._binaries, n)[:n]
        if value:
            return b & members
        return members & ~b

    @staticmethod
    def _post_test(new, old):
        # every bit set in new must also be set in old
        return not bool((new & ~old).any())

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------

    def get_statistics(self, selection, members, get_median_and_mad, add_complement):
        """Sum, sum of squared deviations (and median / MAD for MMAD), the
        same float32 arithmetic as Column.getStatistics() in java."""
        n = self._size
        f = self._floatz[:n]
        if members is None:
            size = n if selection is None else int(np.count_nonzero(selection))
        else:
            size = int(np.count_nonzero(members))

        if not get_median_and_mad:
            sel = selection if members is None else members
            count = _compute_count(sel, f)
            s = _compute_sum(sel, f)
            ssd = _compute_ssd(s, sel, f)
            result = Statistics(count, s, ssd)
        else:
            # TODO (from java) still needs checking for the selection and missing values
            sel = members if members is not None else (selection if selection is not None else np.ones(n, dtype=bool))
            vals = f[sel[:n]]
            s = _seq_sum_f32(vals)
            vals = java_sort_floats(vals)
            median = _compute_median(vals)
            result = Statistics(size, s, _compute_ssd_values(s, vals), median, _compute_mad(median, vals))

        if add_complement:
            if members is None:
                compl = np.zeros(n, dtype=bool)
            else:
                compl = ~members[:n]
            if selection is not None:
                compl = compl & selection[:n]
            c_count = _compute_count(compl, f)
            c_sum = _compute_sum(compl, f)
            result.add_complement(c_count, c_sum, _compute_ssd(c_sum, compl, f))
            # deprecated java method, loops over all values (NaN too)
            mean = np.float32(result.get_average())
            with np.errstate(all='ignore'):
                d = f - mean
                result.add_dataset_ssd(_seq_sum_f32(d * d))
        return result

    def get_domain(self):
        """Sorted distinct values as strings (TreeSet<String>)."""
        t = self._type
        if t == AttributeType.NOMINAL:
            return sorted(set(self._distinct_values))
        if t == AttributeType.NUMERIC:
            # FIXME (java) the strings sort as strings, '10.0' < '2.0'
            return sorted({float_to_string(v) for v in function.get_unique_values(self._floatz)})
        if t == AttributeType.BINARY:
            size = self._get_binaries_cardinality()
            if size == 0:
                return []
            if size == 1:
                # FIXME MM (java) only 1's would need {1}, this gives {}
                if not self._binaries[:self._size].any():
                    return ["0"]
                return []
            if size == 2:
                return ["0", "1"]
            raise AssertionError(size)
        if t == AttributeType.ORDINAL:
            raise AssertionError(t)
        self._log_type_error("Column.getDomain()")
        raise AssertionError(t)

    def build_shared_domain(self):
        self._distinct_values_u = tuple(self._distinct_values)

    def get_shared_domain(self):
        return self._distinct_values_u

    def get_unique_nominal_domain_counts(self, members, members_cardinality):
        """Counts per value code (index = code), the last entry holds the
        number of distinct values present in members."""
        assert members_cardinality > 1
        if self._cardinality == 0:
            return np.array([0], dtype=np.int64)
        if self._cardinality == 1:
            return np.array([members_cardinality, 1], dtype=np.int64)
        codes = self._nominalz[:self._size][members[:self._size]]
        counts = np.zeros(self._cardinality + 1, dtype=np.int64)
        bc = np.bincount(codes, minlength=self._cardinality)
        counts[:self._cardinality] = bc[:self._cardinality]
        counts[self._cardinality] = int(np.count_nonzero(bc))
        return counts

    def get_unique_numeric_domain_map(self, members, members_cardinality=None):
        """NOTE values in itsDomain are unique up to the -1 count sentinel."""
        if not self._is_valid_call("getUniqueNumericDomainMap", members):
            return None
        if members_cardinality is None:
            members_cardinality = int(np.count_nonzero(members))
        vals = java_sort_floats(self._floatz[:self._size][members[:self._size]])
        domain = np.zeros(members_cardinality, dtype=np.float32)
        counts = np.zeros(members_cardinality, dtype=np.int64)
        if vals.size:
            # comparison with != like java, so every NaN is its own value
            new = np.ones(vals.size, dtype=bool)
            new[1:] = vals[1:] != vals[:-1]
            starts = np.flatnonzero(new)
            k = starts.size
            domain[:k] = vals[starts]
            ends = np.append(starts[1:], vals.size)
            counts[:k] = ends - starts
            if k < counts.size:
                counts[k] = -1
        else:
            k = 0
        return DomainMapNumeric(k, members_cardinality, domain, counts)

    def build_sorted(self, target=None):
        """Sorted distinct values plus, per row, the index of its value.
        Rows outside target get the MASK_ON bit (negative index)."""
        self._sorted_floats = function.get_unique_values(self._floatz)
        keys_sorted = _order_keys(self._sorted_floats)
        keys = _order_keys(self._floatz)
        idx = np.searchsorted(keys_sorted, keys).astype(np.int32)
        if target is not None:
            tgt = np.asarray(target[:idx.shape[0]], dtype=bool)
            if tgt.shape[0] < idx.shape[0]:
                tgt = np.concatenate([tgt, np.zeros(idx.shape[0] - tgt.shape[0], dtype=bool)])
            idx = np.where(tgt, idx, idx | MASK_ON).astype(np.int32)
        self._sort_index = idx

    def remove_sorted(self):
        self._sorted_floats = None
        self._sort_index = None

    def get_sorted_value(self, index):
        return float(self._sorted_floats[index])

    def get_sorted_index(self, value):
        from .java_compat import binary_search
        return binary_search(self._sorted_floats, value)

    def get_sorted_values_copy(self):
        return self._sorted_floats.copy()

    def sorted_floats_view(self):
        return self._sorted_floats

    def sort_index_view(self):
        return self._sort_index

    def get_value_count(self, members):
        if not self._is_valid_call("getValueCount", members):
            return ValueCount(np.zeros(0, dtype=np.int64))
        si = self._sort_index[members[:self._size]]
        if si.size and si.min() < 0:
            raise IndexError("ArrayIndexOutOfBoundsException: " + str(int(si.min())))
        return ValueCount(np.bincount(si, minlength=self._sorted_floats.shape[0]).astype(np.int64))

    def get_unique_numeric_domain_map_tp(self, members):
        """ValueCountTP: counts, true positives and missing counts per sorted value."""
        if not self._is_valid_call("getUniqueNumericDomainMap", members):
            return ValueCountTP(np.zeros(0, np.int64), np.zeros(0, np.int64), 0, 0)
        m = self._sorted_floats.shape[0]
        sel = members[:self._size]
        si = self._sort_index[sel]
        miss = self._missing[:self._size][sel]
        nm = ~miss
        pos = si >= 0
        cnt = np.bincount((si[nm] & MASK_OFF).astype(np.int64), minlength=m)[:m].astype(np.int64)
        tps = np.bincount(si[nm & pos].astype(np.int64), minlength=m)[:m].astype(np.int64)
        missing_count = int(np.count_nonzero(miss))
        # TODO (java): this doesn't work correctly
        missing_positive = int(np.count_nonzero(miss & pos))
        return ValueCountTP(cnt, tps, missing_count, missing_positive)

    def get_unique_numeric_domain_map_sum(self, members, target):
        """ValueCountSum, TODO (java) fix for missing values"""
        if not self._is_valid_call("getUniqueNumericDomainMap", members):
            return ValueCountSum(np.zeros(0, np.int64), np.zeros(0))
        m = self._sorted_floats.shape[0]
        sel = members[:self._size]
        si = self._sort_index[sel].astype(np.int64)
        tv = target._floatz[:self._size][sel].astype(np.float64)
        cnt = np.bincount(si, minlength=m)[:m].astype(np.int64)
        sums = np.bincount(si, weights=tv, minlength=m)[:m]
        return ValueCountSum(cnt, sums)

    def get_unique_numeric_domain_map_sq(self, members, target):
        """see SubgroupDiscovery.evaluate_numeric_regular_single_numeric_sum_ssd()"""
        if not self._is_valid_call("getUniqueNumericDomainMap", members):
            return ValueCountSumSquaresSum(np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
        m = self._sorted_floats.shape[0]
        sel = members[:self._size]
        si = self._sort_index[sel].astype(np.int64)
        d = target._floatz[:self._size][sel].astype(np.float64)
        cnt = np.bincount(si, minlength=m)[:m].astype(np.int64)
        sums = np.bincount(si, weights=d, minlength=m)[:m]
        sq = np.bincount(si, weights=d * d, minlength=m)[:m]
        return ValueCountSumSquaresSum(cnt, sums, sq)

    def get_unique_split_points_map(self, members, members_cardinality, nr_splits, operator):
        """deprecated java code kept for the binning strategies, see the long
        comment in Column.java: for <= the split points are inclusive end
        points, for >= inclusive start points."""
        assert operator in (Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL)
        if not self._is_valid_call("getSplitPointsMap", members):
            return DomainMapNumeric(0, 0, np.zeros(0, np.float32), np.zeros(0, np.int64))
        if nr_splits < 0:
            raise ValueError(str(nr_splits) + " (theNrSplits) < 0")
        if nr_splits == 0:
            return DomainMapNumeric(0, 0, np.zeros(0, np.float32), np.zeros(0, np.int64))

        upper = min(members_cardinality, nr_splits)
        if self._cardinality != 0:
            upper = min(upper, self._cardinality)
        d_domain = np.zeros(upper, dtype=np.float32)
        d_counts = np.zeros(upper, dtype=np.int64)

        if members_cardinality == 0:
            return DomainMapNumeric(0, 0, np.zeros(nr_splits, np.float32), np.zeros(nr_splits, np.int64))

        domain = java_sort_floats(self._floatz[:self._size][members[:self._size]])
        dl = [float(x) for x in domain]
        nr_bins = nr_splits + 1
        size = members_cardinality
        idx = -1
        total = 0
        if operator == Operator.LESS_THAN_OR_EQUAL:
            j = 0
            for i in range(1, nr_splits + 1):
                k = (size * i) // nr_bins
                if k < j:
                    continue
                split = dl[k]
                # correct count for the value
                m = k + 1
                while m < size:
                    if float_compare(split, dl[m]) == 0:
                        k += 1
                    m += 1
                idx += 1
                d_domain[idx] = split
                d_counts[idx] = k - j + 1
                j = k + 1
                total = k + 1
        else:
            last = 0.0 if math.isnan(dl[0]) else float('nan')
            for i in range(1, nr_splits + 1):
                k = (size * i) // nr_bins
                split = dl[k]
                if float_compare(split, last) == 0:
                    continue
                last = split
                m = k - 1
                while m >= 0:
                    if float_compare(split, dl[m]) == 0:
                        k -= 1
                    m -= 1
                j = size - k
                idx += 1
                d_domain[idx] = split
                d_counts[idx] = j
                if idx == 0:
                    total = j
                else:
                    d_counts[idx - 1] -= j
        if idx < d_counts.shape[0] - 1:
            d_counts[idx + 1] = -1
        return DomainMapNumeric(idx + 1, total, d_domain, d_counts)

    def get_average(self, selection=None):
        """Average of the non-missing values (float sum, can overflow to NaN)."""
        if self._type != AttributeType.NUMERIC:
            raise ValueError("Invalid call for AttributeType: " + str(self._type))
        n = self._size
        sel = ~self._missing[:n]
        if selection is not None:
            sel = sel & selection[:n]
        vals = self._floatz[:n][sel]
        return fdiv(_seq_sum_f32(vals), int(vals.size))

    def get_average_ranking(self, subgroup):
        from .label_ranking import LabelRanking
        if self._type != AttributeType.NOMINAL:
            self._log_message("getAverageRanking", self._get_type_error("NOMINAL"))
            return None
        result = LabelRanking(self._distinct_values[int(self._nominalz[0])])
        size = result.get_size()
        total = [0] * size
        members = None if subgroup is None else subgroup.get_members()
        cache = {}
        for i in range(self._size):
            if members is None or members[i]:
                code = int(self._nominalz[i])
                r = cache.get(code)
                if r is None:
                    r = LabelRanking(self._distinct_values[code])
                    cache[code] = r
                for j in range(size):
                    total[j] += r.get_rank(j)
        ranks = sorted(total)
        for i in range(size):
            look = total[i]
            first = -1
            for j in range(size):
                if look == ranks[j]:
                    first = j
            # TODO (java) properly deal with ties in the ranking
            result.set_rank(i, first)
        return result

    def get_average_ranking_matrix(self, subgroup):
        from .label_ranking import LabelRanking
        from .label_ranking_matrix import LabelRankingMatrix
        if self._type != AttributeType.NOMINAL:
            self._log_message("getAverageRankingMatrix", self._get_type_error("NOMINAL"))
            return None
        result = LabelRankingMatrix(len(self._distinct_values[int(self._nominalz[0])].replace(">", "")))
        members = None if subgroup is None else subgroup.get_members()
        count = 0
        cache = {}
        for i in range(self._size):
            if members is None or members[i]:
                code = int(self._nominalz[i])
                lrm = cache.get(code)
                if lrm is None:
                    lrm = LabelRankingMatrix(LabelRanking(self._distinct_values[code]))
                    cache[code] = lrm
                result.add(lrm)
                count += 1
        # divide by zero is not possible, subgroups always have members
        result.divide(count)
        return result

    def get_average_ranking_matrix0(self, subgroup):
        from .label_ranking import LabelRanking
        from .label_ranking_matrix import LabelRankingMatrix
        if self._type != AttributeType.NOMINAL:
            self._log_message("getAverageRankingMatrix", self._get_type_error("NOMINAL"))
            return None
        members = None if subgroup is None else subgroup.get_members()
        rank_size = len(self._distinct_values[int(self._nominalz[0])].replace(">", ""))
        lr_mode = LabelRankingMatrix(rank_size)
        mode = np.zeros((3, rank_size, rank_size), dtype=np.int64)
        for k in range(self._size):
            if members is None or members[k]:
                lrm = LabelRankingMatrix(LabelRanking(self._distinct_values[int(self._nominalz[k])]))
                mode += lrm.get_mode_matrix()
        for i in range(rank_size):
            for j in range(rank_size):
                if mode[0, i, j] > mode[1, i, j]:
                    lr_mode.matrix[i, j] = -1 if mode[0, i, j] > mode[2, i, j] else 1
                else:
                    lr_mode.matrix[i, j] = 0 if mode[1, i, j] > mode[2, i, j] else 1
        return lr_mode

    def count_values(self, value, selection=None):
        """Occurrences of value (NOMINAL / BINARY), for BINARY '1' is true."""
        t = self._type
        n = self._size
        if t == AttributeType.NOMINAL:
            v = self._distinct_values_map.get(value)
            if v is None:
                log.log_command_line(str(value) + " does not occur in Column " + self._name)
                return 0
            hit = self._nominalz[:n] == v
            if selection is not None:
                hit = hit & selection[:n]
            return int(np.count_nonzero(hit))
        if t == AttributeType.BINARY:
            b = self._binaries[:n]
            if selection is None:
                c = int(np.count_nonzero(b))
                return c if value == DEFAULT_BINARY_TRUE_STRING else n - c
            if value == DEFAULT_BINARY_TRUE_STRING:
                return int(np.count_nonzero(selection[:n] & b))
            return int(np.count_nonzero(selection[:n] & ~b))
        self._log_message("countValues", self._get_type_error("NOMINAL or BINARY"))
        return 0

    def _is_valid_call(self, source, members):
        err = None
        if self._type not in (AttributeType.NUMERIC, AttributeType.ORDINAL):
            err = self._get_type_error("NUMERIC or ORDINAL")
        if err is not None:
            self._log_message(source, err)
        return err is None

    def _get_type_error(self, valid):
        return "Column can not be of type: %s, must be %s" % (self._type, valid)

    def _log_message(self, source, error):
        log.log_command_line("%s.%s(): %s" % (self._name, source, error))

    def get_subset(self, i):
        """Values whose bits are set in i, bit 0 is the last distinct value.
        No NOMINAL check, use with care."""
        dv = self._distinct_values
        size = len(dv)
        out = []
        for m in range(size):
            if (i >> (size - 1 - m)) & 1:
                out.append(dv[m])
        return out

    # ------------------------------------------------------------------
    # obsolete (deprecated) split point code, still used for bins
    # ------------------------------------------------------------------

    def get_unique_split_points(self, members, nr_splits, operator):
        if not USE_NEW_BINNING:
            return self.get_split_points(members, nr_splits)
        if operator not in (Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL):
            raise AssertionError("FIX new getSplitPoints()")
        if not self._is_valid_call("getSplitPoints", members):
            return np.zeros(0, dtype=np.float32)
        if nr_splits < 0:
            raise ValueError(str(nr_splits) + " (theNrSplits) < 0")
        if nr_splits == 0:
            return np.zeros(0, dtype=np.float32)
        domain = java_sort_floats(self._floatz[:self._size][members[:self._size]])
        size = domain.shape[0]
        if size == 0:
            return np.zeros(0, dtype=np.float32)
        if operator == Operator.LESS_THAN_OR_EQUAL:
            domain = domain[::-1].copy()
        nr_bins = nr_splits + 1
        sp = np.array([domain[(size * (i + 1)) // nr_bins] for i in range(nr_splits)], dtype=np.float32)
        sp = Column._get_unique_values_sorted(sp)
        if operator == Operator.LESS_THAN_OR_EQUAL:
            sp = sp[::-1].copy()
        return sp

    @staticmethod
    def _get_unique_values_sorted(arr):
        """drops consecutive duplicates (array assumed sorted)"""
        if arr.shape[0] == 0:
            return arr.copy()
        out = [arr[0]]
        for v in arr[1:]:
            if out[-1] != v:
                out.append(v)
        return np.array(out, dtype=np.float32)

    def get_split_points(self, members, nr_splits):
        """Equal height bin boundaries (may contain duplicates, see the java
        FIXME: with fewer members than splits the rest are 0.0f)."""
        if not self._is_valid_call("getSplitPoints", members):
            return np.zeros(0, dtype=np.float32)
        if nr_splits < 0:
            raise ValueError(str(nr_splits) + " (theNrSplits) < 0")
        if nr_splits == 0:
            return np.zeros(0, dtype=np.float32)
        sp = np.zeros(nr_splits, dtype=np.float32)
        domain = java_sort_floats(self._floatz[:self._size][members[:self._size]])
        size = domain.shape[0]
        if size == 0:
            return sp
        for j in range(nr_splits):
            # int arithmetic in java, can overflow on huge data
            sp[j] = domain[(size * (j + 1)) // (nr_splits + 1)]
        return sp

    def get_unique_split_points_map_x(self, members, members_cardinality, nr_splits, operator):
        """SortedMap version of get_unique_split_points_map(), returns a
        sorted list of (value, count) pairs."""
        assert operator in (Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL)
        if not self._is_valid_call("getSplitPointsMap", members):
            return []
        if nr_splits < 0:
            raise ValueError(str(nr_splits) + " (theNrSplits) < 0")
        if nr_splits == 0 or members_cardinality == 0:
            return []
        dl = [float(x) for x in java_sort_floats(self._floatz[:self._size][members[:self._size]])]
        size = members_cardinality
        nr_bins = nr_splits + 1
        result = {}
        order = []

        def put(k, v):
            if k not in result:
                order.append(k)
            result[k] = v

        if operator == Operator.LESS_THAN_OR_EQUAL:
            j = 0
            for i in range(1, nr_splits + 1):
                k = (size * i) // nr_bins
                if k < j:
                    continue
                split = dl[k]
                for m in range(k + 1, size):
                    if float_compare(split, dl[m]) == 0:
                        k += 1
                put(split, k - j + 1)
                j = k + 1
        else:
            last = 0.0 if math.isnan(dl[0]) else float('nan')
            for i in range(1, nr_splits + 1):
                k = (size * i) // nr_bins
                split = dl[k]
                if float_compare(split, last) == 0:
                    continue
                for m in range(k - 1, -1, -1):
                    if float_compare(split, dl[m]) == 0:
                        k -= 1
                j = size - k
                put(split, j)
                if i > 1:
                    put(last, result[last] - j)
                last = split
        from functools import cmp_to_key
        keys = sorted(result.keys(), key=cmp_to_key(float_compare))
        return [(k, result[k]) for k in keys]

    def get_unique_split_points_bounded(self, members, members_cardinality, nr_splits):
        """Bounded intervals for the vm-consecutive strategies: sorted list
        of (Interval, count)."""
        from .interval import Interval
        if not self._is_valid_call("getSplitPointsBounded", members):
            return []
        if nr_splits < 0:
            raise ValueError(str(nr_splits) + " (theNrSplits) < 0")
        if nr_splits == 0:
            return []
        # the java method also logs a comparison with the original version,
        # for testing only; just the map version here
        cardinality = int(np.count_nonzero(members))
        mp = self.get_unique_split_points_map_x(members, cardinality, nr_splits, Operator.LESS_THAN_OR_EQUAL)
        if not mp:
            return []
        result = []
        total = 0
        f, i = mp[0]
        total += i
        result.append((Interval(float('-inf'), f), i))
        for k in range(1, len(mp)):
            nv, i = mp[k]
            total += i
            if total == cardinality:
                result.append((Interval(f, float('inf')), i))
            else:
                result.append((Interval(f, nv), i))
            f = nv
        if total != cardinality and float_compare(float('inf'), mp[-1][0]) != 0:
            result.append((Interval(f, float('inf')), cardinality - total))
        result.sort(key=lambda p: p[0])
        return result

    def __repr__(self):
        return "Column(%s, %s)" % (self._name, self._type)


# ----------------------------------------------------------------------
# float32 helpers used by get_statistics()
# ----------------------------------------------------------------------

def _compute_count(sel, f):
    if sel is None:
        return int(np.count_nonzero(~np.isnan(f)))
    v = f[sel[:f.shape[0]]]
    return int(np.count_nonzero(~np.isnan(v)))


def _compute_sum(sel, f):
    v = f if sel is None else f[sel[:f.shape[0]]]
    return _seq_sum_f32(v[~np.isnan(v)])


def _compute_ssd(the_sum, sel, f):
    """recomputes sum and count itself, like the java version"""
    v = f if sel is None else f[sel[:f.shape[0]]]
    v = v[~np.isnan(v)]
    s = _seq_sum_f32(v)
    mean = np.float32(fdiv(s, int(v.size)))
    with np.errstate(all='ignore'):
        d = v - mean
        return _seq_sum_f32(d * d)


def _compute_ssd_values(the_sum, vals):
    mean = np.float32(fdiv(the_sum, int(vals.shape[0])))
    with np.errstate(all='ignore'):
        d = vals - mean
        return _seq_sum_f32(d * d)


def _compute_median(sorted_vals):
    n = sorted_vals.shape[0]
    if n == 0:
        raise IndexError("ArrayIndexOutOfBoundsException")
    if (n & 1) == 0:
        return float(np.float32(np.float32(sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / np.float32(2)))
    return float(sorted_vals[n // 2])


def _compute_mad(median, sorted_vals):
    with np.errstate(all='ignore'):
        d = np.abs(sorted_vals - np.float32(median)).astype(np.float32)
    d = java_sort_floats(d)
    return _compute_median(d)
