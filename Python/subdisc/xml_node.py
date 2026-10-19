"""Convenience functions for building xml nodes (java XMLNode)."""

import xml.etree.ElementTree as ET

from .java_compat import jstr


def add_node_to(parent, element_name, text_content=None, *, empty=False):
    """Adds a child element (name lowercased) and returns it.

    Without text_content the element stays empty, like the 2-argument java
    method. Anything else is written with its java toString() text.
    """
    node = ET.SubElement(parent, element_name.lower())
    if text_content is not None:
        node.text = text_content if isinstance(text_content, str) else jstr(text_content)
    elif empty:
        node.text = ""
    return node


def text_of(node):
    """getTextContent() of an element (all text inside it)."""
    return "".join(node.itertext())
