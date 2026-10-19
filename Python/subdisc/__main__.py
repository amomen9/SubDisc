"""python -m subdisc [file.xml [showWindows [nrThreads]]]

An autorun without showWindows does not need Qt at all, so it also works
on a machine without PySide6.
"""

import sys


def main(args=None):
    args = sys.argv[1:] if args is None else list(args)
    if args:
        from .attribute_type import AttributeType
        show_windows = len(args) >= 2 and AttributeType.is_valid_binary_true_value(args[1])
        if not show_windows:
            from .xml_auto_run import auto_run_setting
            auto_run_setting(args)
            return 0
    from .gui.app import main as gui_main
    return gui_main(args)


if __name__ == "__main__":
    sys.exit(main())
