"""Directed acyclic graph of the Bayesian network, with its NetworkNodes."""

from . import log
from .cross_cube import CrossCube
from .item_set import ItemSet


class NetworkNode:
    __slots__ = ("_parameters", "_parents", "_children", "_quality", "_name")

    def __init__(self, parameters, size, name):
        self._parameters = parameters
        self._children = ItemSet(size)
        self._parents = ItemSet(size)
        self._quality = -100000.0
        self._name = name

    def add_child(self, child):
        self._children.set(child)

    def remove_child(self, child):
        self._children.clear(child)

    def remove_parent(self, parent):
        self._parents.clear(parent)

    def add_parent(self, parent):
        self._parents.set(parent)

    def is_this_my_child(self, node):
        return self._children.get(node)

    def is_this_my_parent(self, node):
        return self._parents.get(node)

    def is_connected(self, node):
        """1: node is my parent, 2: node is my child, 0: not connected"""
        if self._parents.get(node):
            return 1
        if self._children.get(node):
            return 2
        return 0

    def get_nr_parents(self):
        return self._parents.get_item_count()

    def get_parents(self):
        return self._parents

    def get_nr_children(self):
        return self._children.get_item_count()

    def get_children(self):
        return self._children

    def get_quality(self):
        return self._quality

    def set_quality(self, q):
        self._quality = q

    def get_parameters(self):
        return self._parameters

    def set_parameters(self, p):
        self._parameters = p

    def get_name(self):
        return self._name


class DAG:
    def __init__(self, size_or_targets):
        self._quality = 0.0
        self._no_arcs = 0
        if isinstance(size_or_targets, int):
            n = size_or_targets
            self._size = n
            self._nodes = []
            for i in range(n):
                self._nodes.append(NetworkNode(None, n, "node " + str(i + 1)))
                self._quality += self._nodes[i].get_quality()
        else:
            targets = list(size_or_targets)
            n = len(targets)
            self._size = n
            self._nodes = []
            for c in targets:
                node = NetworkNode(None, n, c.get_name())
                self._nodes.append(node)
                self._quality += node.get_quality()
        self._visitation = [0] * self._size

    def print(self):
        """nodes are printed 1..n"""
        for i in range(self._size):
            for j in range(i):
                c = self.get_node(j).is_connected(i)
                if c == 1:
                    log.log_command_line("%d -> %d" % (i + 1, j + 1))
                elif c == 2:
                    log.log_command_line("%d -> %d" % (j + 1, i + 1))

    def get_size(self):
        return self._size

    def get_node(self, nd):
        return self._nodes[nd]

    def get_name(self, node):
        return self._nodes[node].get_name()

    def add_arc(self, from_node, to_node, only_add):
        self.get_node(from_node).add_child(to_node)
        self.get_node(to_node).add_parent(from_node)
        self._no_arcs += 1
        if not only_add:
            self.fix_dimensions(to_node)

    def remove_arc(self, from_node, to_node, only_remove):
        if not self.get_node(from_node).is_this_my_child(to_node):
            return False
        self.get_node(from_node).remove_child(to_node)
        self.get_node(to_node).remove_parent(from_node)
        self._no_arcs -= 1
        if not only_remove:
            self.fix_dimensions(to_node)
        return True

    def add_arc_acyclic(self, from_node, to_node, only_add):
        self.get_node(from_node).add_child(to_node)
        self.get_node(to_node).add_parent(from_node)
        self._visitation = [0] * self._size
        self._visitation[from_node] = 1
        cyclic = self._cyclic_test(self.get_node(from_node).get_children())
        if cyclic:
            self.get_node(from_node).remove_child(to_node)
            self.get_node(to_node).remove_parent(from_node)
        else:
            self._no_arcs += 1
            if not only_add:
                self.fix_dimensions(to_node)
        return not cyclic

    def _cyclic_test(self, children):
        for l in children.set_bits():
            if self._visitation[l] == 2:
                continue   # node already closed
            if self._visitation[l] == 1:
                return True   # visited before: cycle
            self._visitation[l] = 1
            if self._cyclic_test(self.get_node(l).get_children()):
                return True
            self._visitation[l] = 2
        return False

    def get_quality(self, node=None):
        if node is None:
            return self._quality
        return self.get_node(node).get_quality()

    def set_quality(self, node, quality):
        n = self.get_node(node)
        self._quality = self._quality - n.get_quality() + quality
        n.set_quality(quality)

    def fix_dimensions(self, nd):
        self.get_node(nd).set_parameters(CrossCube(self.get_node(nd).get_nr_parents()))

    def determine_v_structures(self):
        """only the edge missing in a v-structure is stored (enough here)"""
        n = self._size
        result = [[False] * n for _ in range(n)]
        for x in range(n):
            for y in range(x + 1, n):
                c = self.get_node(y).is_connected(x)
                if c == 0:
                    for z in range(y + 1, n):
                        if self.get_node(z).is_connected(x) == 1 and self.get_node(z).is_connected(y) == 1:
                            result[x][y] = True
                elif c == 1:
                    for z in range(y + 1, n):
                        if self.get_node(z).is_connected(x) == 0 and self.get_node(z).is_connected(y) == 2:
                            result[x][z] = True
                elif c == 2:
                    for z in range(y + 1, n):
                        if self.get_node(z).is_connected(x) == 2 and self.get_node(z).is_connected(y) == 0:
                            result[y][z] = True
        return result

    def test_v_structure(self, x, y):
        c = self.get_node(y).is_connected(x)
        n = self._size
        if c == 0:
            for z in range(n):
                if self.get_node(z).is_connected(x) == 1 and self.get_node(z).is_connected(y) == 1:
                    return True
        elif c == 1:
            for z in range(n):
                if self.get_node(z).is_connected(x) == 0 and self.get_node(z).is_connected(y) == 2:
                    return True
        elif c == 2:
            for z in range(y + 1, n):
                if self.get_node(z).is_connected(x) == 2 and self.get_node(z).is_connected(y) == 0:
                    return True
        return False
