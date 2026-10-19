"""SubDisc: Subgroup Discovery (Python port of the java SubDisc / Cortana tool).

The mining engine lives in this package, the PySide6 gui in subdisc.gui.
Run the gui with:  python -m subdisc
"""

import numpy as _np

# java float / double arithmetic gives inf and nan silently, numpy should too
_np.seterr(all='ignore')

__version__ = "2.1142"
