"""Reads a beam seed (one ConditionList per line) as written by CFTP.

Example lines, no AND between the conditions, just a space, and a space at
the end of the line:
    'spectacle-prescrip=myope age=presbyopic '
    'age=young tear-prod-rate=reduced spectacle-prescrip=hypermetrope '
"""

import traceback

from . import log
from . import condition_list_builder as clb
from .attribute_type import AttributeType
from .condition import Condition
from .condition_base import ConditionBase
from .operators import Operator


class LoaderFraunhofer:
    def __init__(self, path, table, search_parameters):
        self._beam_seed = []
        self._table = table
        self._search_parameters = search_parameters
        try:
            with open(path, encoding="utf-8") as f:
                nr = 0
                for line in f:
                    nr += 1
                    self._beam_seed.append(self._convert(line.rstrip("\r\n")))
                    if nr % 100 == 0:
                        log.log_command_line(str(nr) + " lines read")
            search_parameters.set_beam_seed(self._beam_seed)
        except OSError:
            traceback.print_exc()

    def _convert(self, s):
        # java String.split drops trailing empty strings, so the space at
        # EOL is fine
        parts = s.split(" ")
        while parts and parts[-1] == "":
            parts.pop()
        cl = clb.empty_list()
        for atom in parts:
            # only equality conditions, no '=' in names or values
            ref = atom.split("=")
            col = self._table.get_column(ref[0])
            b = ConditionBase(col, Operator.EQUALS)
            value = ref[1]
            t = col.get_type()
            if t == AttributeType.NOMINAL:
                cond = Condition(b, value)
            elif t == AttributeType.BINARY:
                if not AttributeType.is_valid_binary_value(value):
                    raise ValueError(value + " is not a valid BINARY value")
                cond = Condition(b, AttributeType.is_valid_binary_true_value(value))
            else:
                raise AssertionError(t)
            cl = clb.create_list(cl, cond)
        return cl
