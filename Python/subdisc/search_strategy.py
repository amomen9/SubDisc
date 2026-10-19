from enum import Enum

from . import log


class SearchStrategy(Enum):
    # (ordinal, gui text, is beam, requires search width parameter)
    BEAM = (0, "beam", True, True)
    ROC_BEAM = (1, "ROC beam", True, False)
    COVER_BASED_BEAM_SELECTION = (2, "cover-based beam selection", True, True)
    BEST_FIRST = (3, "best first", False, False)
    DEPTH_FIRST = (4, "depth first", False, False)
    BREADTH_FIRST = (5, "breadth first", False, False)

    @property
    def GUI_TEXT(self):
        return self.value[1]

    def __str__(self):
        return self.value[1]

    @staticmethod
    def from_string(text):
        for s in SearchStrategy:
            if s.GUI_TEXT.lower() == (text or "").lower():
                return s
        log.log_command_line("'%s' is not a valid SearchStrategy. Returning '%s'." % (text, SearchStrategy.get_default().GUI_TEXT))
        return SearchStrategy.get_default()

    @staticmethod
    def get_default():
        return SearchStrategy.BEAM

    def is_beam(self):
        return self.value[2]

    def requires_search_width_parameter(self):
        """Beam strategies need a width, exhaustive ones do not."""
        return self.value[3]
