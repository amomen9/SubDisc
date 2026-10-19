"""Adds a CUI domain (association scores per concept) to a loaded gene table.

NOTE this may need a lot of memory, the domain files are big.
"""

import os
import time

from . import cui
from . import log
from . import ui_hooks
from .attribute_type import AttributeType
from .column import Column
from .error_log import ErrorLog
from .file_type import FileType
from .java_compat import parse_float, to_f32

# TODO (java) should come from the gui, values below it stay 0.0f
THRESHOLD = to_f32(0.01)

_cui2line_nr_map = None   # static in java
_cui2name_map = None


def _id_string(s):
    """String.valueOf(Float.valueOf(s).intValue()), removes the '.0'"""
    f = parse_float(s)
    if f != f:
        return "0"
    return str(int(max(min(f, 2147483647.0), -2147483648.0)))


class FileLoaderGeneRank:
    def __init__(self, table, enrichment_type, domain_file=None):
        self._table = table
        self._gene2cui_map = None
        self._source = None
        if table is None or enrichment_type is None:
            log.log_command_line("FileLoaderRank Constructor: parameter(s) can not be 'null'.")
            return
        self._check_enrichment_type(enrichment_type, domain_file)
        if self._source is None:
            log.log_command_line("No enrichment source file selected.")
        elif not os.path.exists(self._source):
            ErrorLog.log(self._source, FileNotFoundError())
        elif os.path.getsize(self._source) == 0:
            log.log_command_line("Empty file: " + self._source)
        else:
            start = time.time()
            self._foo()
            print("%ds. for loading of domain file." % int(time.time() - start))

    def _check_enrichment_type(self, t, domain_file):
        if t == cui.EnrichmentType.CUI:
            self._setup_cui_maps(domain_file)
        elif t in (cui.EnrichmentType.GO, cui.EnrichmentType.CUSTOM):
            pass   # empty in java as well
        else:
            log.log_command_line("Unknown EnrichmentType: " + str(t))

    def _setup_cui_maps(self, domain_file):
        global _cui2line_nr_map, _cui2name_map
        lnr = cui.GENE_IDENTIFIER_CUIS
        name = cui.CUI2NAME
        if not os.path.exists(lnr):
            ErrorLog.log(lnr, FileNotFoundError())
        elif not os.path.exists(name):
            ErrorLog.log(name, FileNotFoundError())
        else:
            _cui2line_nr_map = cui.cui2line_nr_map(lnr)
            _cui2name_map = cui.cui2name_map(name)
        if _cui2line_nr_map is None:
            log.log_command_line("File: '%s' not found." % cui.GENE_IDENTIFIER_CUIS)
            return
        if domain_file is not None:
            self._source = domain_file
        else:
            chooser = getattr(ui_hooks, "choose_cui_domain", None)
            self._source = chooser(cui.available_domains()) if chooser else None

    def _foo(self):
        try:
            f = open(self._source, encoding="utf-8", newline="")
        except OSError:
            return
        with f:
            has_cui_column = False
            has_rank_column = False
            identifier_column = -1
            columns = self._table.get_columns()
            nr_data_columns = len(columns)
            nr_rows = self._table.get_nr_rows()
            header = f.readline().rstrip("\r\n").split("\t")
            nr_cui_columns = len(header)

            for i in range(nr_data_columns):
                s = columns[i].get_name().lower()
                if s == "rank":
                    has_rank_column = True
                elif s in ("entrez", "go"):
                    self._create_gene2cui_map(s)
                    identifier_column = i
                elif s == "cui":   # no mapping needed
                    has_cui_column = True
                    identifier_column = i
                if has_rank_column and identifier_column != -1:
                    break

            if identifier_column == -1:
                chooser = getattr(ui_hooks, "choose_identifier", None)
                res = chooser(columns) if chooser else None
                if res is None:
                    log.log_command_line("No identifier column selected.")
                    return
                identifier_column, id_type = res
                if identifier_column == -1 or id_type is None:
                    return
                self._create_gene2cui_map(id_type)

            domain_name = FileType.remove_extension(self._source)
            if not self._table.add_domain(domain_name):
                return

            columns[identifier_column].set_is_enabled(False)   # disable by default

            if not has_rank_column:
                rank = Column("RANK", None, AttributeType.NUMERIC, nr_data_columns, nr_rows)
                nr_data_columns += 1
                for r in range(1, nr_rows + 1):
                    rank.add_float(float(r))
                rank.set_is_enabled(False)   # XXX for now disable by default
                columns.append(rank)

            columns.append(Column("Domain: " + domain_name, None, AttributeType.NOMINAL, nr_data_columns, nr_rows))
            nr_data_columns += 1
            columns[nr_data_columns - 1].set_is_enabled(False)

            for i in range(1, nr_cui_columns):
                # NOTE browse/result window use the normal name, not the short one
                nm = _cui2name_map.get(header[i]) if _cui2name_map else None
                columns.append(Column(nm if nm is not None else "null", header[i], AttributeType.NUMERIC, nr_data_columns, nr_rows))
                nr_data_columns += 1

            line_nr_map = {}
            id_column = columns[identifier_column]
            nr_columns = len(self._table.get_columns())
            domain_column = self._table.get_column(nr_columns - nr_cui_columns)
            cui_cols = [self._table.get_column(k) for k in range(nr_columns - nr_cui_columns + 1, nr_columns)]

            for i in range(nr_rows):
                # all association scores start at 0.0f, cuis are set later
                for c in cui_cols:
                    c.add_float(0.0)
                key = _id_string(id_column.get_string(i))
                c_ui = key if has_cui_column else (self._gene2cui_map or {}).get(key)
                line_nr = _cui2line_nr_map.get(c_ui) if c_ui is not None else None
                nm = _cui2name_map.get(c_ui) if (_cui2name_map and c_ui is not None) else None
                # java adds a null String here, it shows as 'null'
                domain_column.add_nominal(nm if nm is not None else "null")
                # no mapping may exist in some ENTREZ/GO cases
                if line_nr is not None:
                    line_nr_map[line_nr] = i

            # read the domain file in one pass, line numbers in order
            current = 1   # header line read already
            for needed in sorted(line_nr_map):
                row = line_nr_map[needed]
                current += 1
                while current <= needed:
                    f.readline()
                    current += 1
                a = f.readline().rstrip("\r\n").split("\t")
                # THE CRUCIAL THRESHOLD CHECK (java TODO, should come from gui)
                for j in range(1, nr_cui_columns):
                    v = parse_float(a[j])
                    if v > THRESHOLD:
                        cui_cols[j - 1].set(row, v)
            for c in columns[-(nr_cui_columns + (0 if has_rank_column else 1)):]:
                c.close()

    def _create_gene2cui_map(self, identifier_type):
        if identifier_type == "entrez":
            self._gene2cui_map = cui.Gene2CuiMap.ENTREZ2CUI.get_map()
        elif identifier_type.lower() == "go":
            self._gene2cui_map = cui.Gene2CuiMap.GO2CUI.get_map()

    def get_table(self):
        return self._table
