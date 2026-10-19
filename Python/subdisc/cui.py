"""CUI (concept unique identifier) maps for the gene rank enrichment, java
package nl.liacs.subdisc.cui. All paths are relative to the current
directory, like in java."""

import enum
import os

from .error_log import ErrorLog

CUI_DIR = "CUI/"
CUI2NAME = "CUI/conceptID2name.txt"
GENE_IDENTIFIER_CUIS = "CUI/gene_identifier_cuis.txt"
ENTREZ2CUI = "CUI/entrezGeneToConceptID.txt"
GO2CUI = "CUI/GO2conceptID.txt"
DOMAIN_FILE_PREFIX = "expr2"

NR_DOMAINS = 28
NR_CUI = 314519
NR_EXPRESSION_CUI = 23218   # many entrez/go map to one cui
NR_ENTREZ_CUI = 64870
NR_GO_CUI = 15879

# TODO (java) discuss with CMSB to find another way (eg. release.info file)
KNOWN_DOMAINS = {
    "Acquired Abnormality", "Anatomical Abnormality", "Anatomical Structure",
    "Anatomy", "Biological Process", "Body Location or Region",
    "Body Part, Organ, or Organ Component", "Body Space or Junction",
    "Body Substance", "Body System", "Cell Component",
    "Cell or Molecular Dysfunction", "Cell", "Cellular Component",
    "Congenital Abnormality", "Disease or Syndrome", "Disorders", "drugbank",
    "Embryonic Structure", "Fully Formed Anatomical Structure", "Genes", "GO",
    "Mental or Behavioral Dysfunction", "Molecular Function",
    "Neoplastic Process", "Pathologic Function", "Sign or Symptom", "Tissue",
}


class EnrichmentType(enum.Enum):
    CUI = 0
    GO = 1
    CUSTOM = 2


def _read_lines(path):
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            yield line.rstrip("\r\n")


def cui2line_nr_map(path):
    """line (cui) -> 1 based line number, or None when the file is missing"""
    if path is None or not os.path.exists(path):
        ErrorLog.log(path, FileNotFoundError())
        return None
    out = {}
    try:
        for nr, line in enumerate(_read_lines(path), 1):
            out[line] = nr
    except OSError as e:
        ErrorLog.log(path, e)
    return out


def cui2name_map(path):
    if path is None or not os.path.exists(path):
        ErrorLog.log(path, FileNotFoundError())
        return None
    out = {}
    try:
        for line in _read_lines(path):   # there is no header line
            a = line.split("\t")
            out[a[0]] = a[1]
    except OSError as e:
        ErrorLog.log(path, e)
    return out


class Gene2CuiMap(enum.Enum):
    ENTREZ2CUI = ENTREZ2CUI
    GO2CUI = GO2CUI

    def get_map(self):
        # java builds these when the enum class loads, here on first use
        cache = _GENE2CUI_CACHE
        if self not in cache:
            path = self.value
            if not os.path.exists(path):
                ErrorLog.log(path, FileNotFoundError())
                cache[self] = None
            else:
                m = {}
                try:
                    it = _read_lines(path)
                    next(it, None)   # skip header line
                    for line in it:
                        a = line.split("\t")
                        m[a[0]] = a[1]
                except OSError as e:
                    ErrorLog.log(path, e)
                cache[self] = m
        return cache[self]


_GENE2CUI_CACHE = {}


def available_domains():
    """domain files in CUI_DIR whose name (without extension) is known, sorted"""
    if not os.path.exists(CUI_DIR):
        ErrorLog.log(CUI_DIR, FileNotFoundError())
        return []
    out = []
    for f in os.listdir(CUI_DIR):
        p = os.path.join(CUI_DIR, f)
        i = f.rfind('.')
        if os.path.isfile(p) and (f if i < 0 else f[:i]) in KNOWN_DOMAINS:
            out.append(os.path.abspath(p))
    return sorted(out)


def create_gene_identifier_cuis(path):
    """writes gene_identifier_cuis.txt next to a domain file (first field of
    every line after the header). Not part of the public api in java."""
    if path is None or not os.path.exists(path):
        ErrorLog.log(path, FileNotFoundError())
        return
    lst = []
    try:
        it = _read_lines(path)
        next(it, None)
        for line in it:
            lst.append(line.split(",", 1)[0])
    except OSError as e:
        ErrorLog.log(path, e)
        return
    out = os.path.dirname(path) + GENE_IDENTIFIER_CUIS[3:]
    try:
        with open(out, "w", encoding="utf-8", newline="") as w:
            for s in lst:
                w.write(s)
                w.write("\n")
    except OSError as e:
        ErrorLog.log(path, e)
