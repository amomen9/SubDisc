"""Starts SubDisc, same as 'python -m subdisc'.

python SubDisc.py                          main window
python SubDisc.py file.xml [true] [n]      autorun of an xml experiment file
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subdisc.__main__ import main   # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
