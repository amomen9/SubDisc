"""What to mine for: the target type plus its target column(s).

The target type is never None (SINGLE_NOMINAL by default).
"""

from .java_compat import parse_int
from .target_type import TargetType
from .xml_node import add_node_to, text_of


class TargetConcept:
    # when adding / removing members update add_node_to() and from_xml_node()
    def __init__(self):
        # FIXME MM (java) confusing member, never used in code, always 1
        self._nr_target_attributes = 1
        self._target_type = TargetType.get_default()
        self._primary_target = None
        self._target_value = None
        self._secondary_target = None
        self._multi_regression_targets = None
        self._multi_targets = None
        # double regression code
        self._secondary_targets = None
        self._tertiary_targets = None
        self._intercept_relevance = False
        self._global_regression_model = None

    @staticmethod
    def from_xml_node(node, table):
        """needs the Table to be loaded first"""
        tc = TargetConcept()
        tc._target_type = None
        for child in node:
            tag = child.tag.lower()
            text = text_of(child)
            if tag == "nr_target_attributes":
                tc._nr_target_attributes = parse_int(text)
            if tag == "target_type":
                tc._target_type = TargetType.from_string(text)
            elif tag == "primary_target":
                tc._primary_target = table.get_column(text)
            elif tag == "target_value":
                tc._target_value = text
            elif tag == "secondary_target":
                tc._secondary_target = table.get_column(text)
            elif tag == "multi_targets":
                if text != "":
                    tc._multi_targets = [table.get_column(s) for s in text.split(",")]
            elif tag == "multi_regression_targets":
                # TODO MM (java) these are not present in the XML
                if text != "":
                    tc._secondary_targets = [table.get_column(s) for s in text.split(",")]
                    tc._tertiary_targets = []
        if tc._target_type is None:
            tc._target_type = TargetType.get_default()
        return tc

    def get_nr_target_attributes(self):
        return self._nr_target_attributes

    def set_nr_target_attributes(self, n):
        self._nr_target_attributes = n

    def get_target_type(self):
        return self._target_type

    def set_target_type(self, t):
        self._target_type = TargetType.from_string(t) if isinstance(t, str) else t

    def get_primary_target(self):
        return self._primary_target

    def set_primary_target(self, c):
        self._primary_target = c

    def get_target_value(self):
        return self._target_value

    def set_target_value(self, v):
        self._target_value = v

    def get_secondary_target(self):
        return self._secondary_target

    def set_secondary_target(self, c):
        self._secondary_target = c

    def get_multi_targets(self):
        return self._multi_targets

    def set_multi_targets(self, lst):
        self._multi_targets = lst

    def get_multi_regression_targets(self):
        return self._multi_regression_targets

    def set_multi_regression_targets(self, lst):
        self._multi_regression_targets = lst

    def get_secondary_targets(self):
        return self._secondary_targets

    def set_secondary_targets(self, lst):
        self._secondary_targets = lst

    def get_tertiary_targets(self):
        return self._tertiary_targets

    def set_tertiary_targets(self, lst):
        self._tertiary_targets = lst

    def get_nr_targets(self):
        return 1 + len(self._secondary_targets) + len(self._tertiary_targets)

    def get_intercept_relevance(self):
        return self._intercept_relevance

    def set_intercept_relevance(self, v):
        self._intercept_relevance = v

    def update_to_new_table(self, table):
        """Point at a copy of the Table (CrossValidation makes one per fold)."""
        if self._primary_target is not None:
            self._primary_target = table.get_column(self._primary_target.get_index())
        if self._secondary_target is not None:
            self._secondary_target = table.get_column(self._secondary_target.get_index())
        if self._multi_targets is not None:
            # sic, java copies the old list and then appends the new columns
            lst = list(self._multi_targets)
            for c in self._multi_targets:
                lst.append(table.get_column(c.get_index()))
            self._multi_targets = lst
        # TODO (java) tertiary targets and multi-regression, when stable

    def is_target_attribute(self, column):
        t = self._target_type
        if t in (TargetType.SINGLE_NOMINAL, TargetType.SINGLE_NUMERIC, TargetType.LABEL_RANKING):
            return self._primary_target is column
        if t in (TargetType.MULTI_NUMERIC, TargetType.MULTI_LABEL):
            return any(c is column for c in self._multi_targets)
        if t in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY, TargetType.SCAPE):
            return self._primary_target is column or self._secondary_target is column
        raise AssertionError(t)

    def get_global_regression_model(self):
        return self._global_regression_model

    def set_global_regression_model(self, m):
        self._global_regression_model = m

    def add_node_to(self, parent):
        node = add_node_to(parent, "target_concept")
        add_node_to(node, "nr_target_attributes", self._nr_target_attributes)
        add_node_to(node, "target_type", self._target_type.GUI_TEXT)
        if self._primary_target is None:
            add_node_to(node, "primary_target")
        else:
            add_node_to(node, "primary_target", self._primary_target.get_name())
        add_node_to(node, "target_value", "" if self._target_value is None else self._target_value)
        if self._secondary_target is None:
            add_node_to(node, "secondary_target")
        else:
            add_node_to(node, "secondary_target", self._secondary_target.get_name())
        if not self._multi_targets:
            add_node_to(node, "multi_targets")
        else:
            add_node_to(node, "multi_targets", ",".join(c.get_name() for c in self._multi_targets))
        # multi_regression_targets / secondary_targets / tertiary_targets are
        # disabled in the java version for the stable jar

    def __str__(self):
        from .search_parameters import add_line
        sb = ["target_concept\n"]
        add_line(sb, "nr_target_attributes", str(self._nr_target_attributes))
        add_line(sb, "target_type", self._target_type.GUI_TEXT)
        if self._primary_target is not None:
            add_line(sb, "primary_target", self._primary_target.get_name())
        add_line(sb, "target_value", "null" if self._target_value is None else self._target_value)
        if self._secondary_target is not None:
            add_line(sb, "secondary_target", self._secondary_target.get_name())
        if self._multi_targets:
            add_line(sb, "multi_targets", ", ".join(c.get_name() for c in self._multi_targets))
        return "".join(sb)
