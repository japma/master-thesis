from itertools import count

import networkx as nx
import numpy as np
from networkx.classes.digraph import DiGraph


class EiNetAddress:
    """
    Address of a PC node to its EiNet implementation.

    In EiNets, each layer implements a tensor of log-densities of shape
        (batch_size, vector_length, num_nodes)
    All DistributionVector's, which are either vectors of leaf distributions (exponential families) or vectors of
    sum nodes, uniquely correspond to some slice of the log-density tensor of some layer, where we slice the last axis.

    EiNetAddress stores the "address" of the implementation in the EinsumNetwork.
    """

    def __init__(self, layer=None, idx=None, replica_idx=None) -> None:
        """
        :param layer: which layer implements this node?
        :param idx: which index does the node have in the the layers log-density tensor?
        :param replica_idx: this is solely for the input layer -- see ExponentialFamilyArray and FactorizedLeafLayer.
                            These two layers implement all leaves in parallel. To this end we need "enough leaves",
                            which is achieved to make a sufficiently large "block" of input distributions.
                            The replica_idx indicates in which slice of the ExponentialFamilyArray a leaf is
                            represented.
        """
        self.layer = layer
        self.idx = idx
        self.replica_idx = replica_idx


class DistributionVector:
    """
    Represents either a vectorized leaf or a vectorized sum node in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    # we assign each object a unique id.
    _id_counter = count(0)

    def __init__(self, scope) -> None:
        """
        :param scope: the scope of this node
        """
        self.scope = tuple(sorted(scope))
        self.num_dist = None
        self.einet_address = EiNetAddress()
        self.id = next(self._id_counter)

    def __lt__(self, other) -> bool:
        if isinstance(other, Product):
            return True
        else:
            return (self.scope, self.id) < (other.scope, other.id)


class Product:
    """
    Represents a (cross-)product in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    # we assign each object a unique id.
    _id_counter = count(0)

    def __init__(self, scope) -> None:
        self.scope = tuple(sorted(scope))
        self.id = next(self._id_counter)

    def __lt__(self, other) -> bool:
        if isinstance(other, DistributionVector):
            return False
        else:
            return (self.scope, self.id) < (other.scope, other.id)


def check_if_is_partition(X, P) -> bool:
    """
    Checks if P represents a partition of X.

    :param X: some iterable representing a set of objects.
    :param P: some iterable of iterables, representing a set of sets.
    :return: True of P is a partition of X
                 i) union over P is X
                 ii) sets in P are non-overlapping
    """
    P_as_sets = [set(p) for p in P]
    union = set().union(*[set(p) for p in P_as_sets])
    non_overlapping = len(union) == sum([len(p) for p in P_as_sets])
    return set(X) == union and non_overlapping


def check_graph(graph: DiGraph[DistributionVector]) -> tuple[bool, LiteralString]:
    """
    Check if a graph satisfies our requirements for PC graphs.

    :param graph:
    :return: True/False (bool), string description
    """

    contains_only_PC_nodes = all(
        isinstance(n, (DistributionVector, Product)) for n in graph.nodes()
    )

    is_DAG = nx.is_directed_acyclic_graph(graph)
    is_connected = nx.is_connected(graph.to_undirected())

    sums = get_sums(graph)
    products = get_products(graph)

    products_one_parents = all(len(list(graph.predecessors(p))) == 1 for p in products)
    products_two_children = all(len(list(graph.successors(p))) == 2 for p in products)

    sum_to_products = all(
        all(isinstance(p, Product) for p in graph.successors(s)) for s in sums
    )
    product_to_dist = all(
        all(isinstance(s, DistributionVector) for s in graph.successors(p))
        for p in products
    )
    alternating = sum_to_products and product_to_dist

    proper_scope = all(len(n.scope) == len(set(n.scope)) for n in graph.nodes())
    smooth = all(all(p.scope == s.scope for p in graph.successors(s)) for s in sums)
    decomposable = all(
        check_if_is_partition(p.scope, [s.scope for s in graph.successors(p)])
        for p in products
    )

    check_passed = (
        contains_only_PC_nodes
        and is_DAG
        and is_connected
        and products_one_parents
        and products_two_children
        and alternating
        and proper_scope
        and smooth
        and decomposable
    )

    msg = ""
    if check_passed:
        msg += "Graph check passed.\n"
    if not contains_only_PC_nodes:
        msg += "Graph does not only contain DistributionVector or Product nodes.\n"
    if not is_connected:
        msg += "Graph not connected.\n"
    if not products_one_parents:
        msg += "Products do not have exactly one parent.\n"
    if not products_two_children:
        msg += "Products do not have exactly two children.\n"
    if not alternating:
        msg += "Graph not alternating.\n"
    if not proper_scope:
        msg += "Scope is not proper.\n"
    if not smooth:
        msg += "Graph is not smooth.\n"
    if not decomposable:
        msg += "Graph is not decomposable.\n"

    return check_passed, msg.rstrip()


def get_roots(graph):
    return [n for n, d in graph.in_degree() if d == 0]


def get_sums(graph) -> list[DistributionVector]:
    return [
        n for n, d in graph.out_degree() if d > 0 and isinstance(n, DistributionVector)
    ]


def get_products(graph) -> list[Product]:
    return [n for n in graph.nodes() if isinstance(n, Product)]


def get_leaves(graph):
    return [n for n, d in graph.out_degree() if d == 0]


def partition_on_node(graph, node, scope_partition) -> tuple[Product, list[DistributionVector]]:
    """
    Helper routine to extend the graph.

    Takes a node and adds a new product child to it. Furthermore, as children of the product, it adds new
    DistributionVector nodes with scopes as prescribed in scope_partition (must be a proper partition of the node's
    scope).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param scope_partition: partition of the node's scope
    :return: the product and a list if the product's children
    """

    if not check_if_is_partition(node.scope, scope_partition):
        raise AssertionError("Not a partition.")

    product = Product(node.scope)
    graph.add_edge(node, product)
    product_children = [DistributionVector(scope) for scope in scope_partition]
    for c in product_children:
        graph.add_edge(product, c)

    return product, product_children


def randomly_partition_on_node(
    graph: DiGraph[DistributionVector], node, num_parts: int=2, proportions=None, rand_state=None
) -> tuple[Product, list[DistributionVector]]:
    """
    Calls partition_on_node with a random partition -- used for random binary trees (RAT-SPNs).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param num_parts: number of parts in the partition (int)
    :param proportions: split proportions (list of numbers)
    :param rand_state: numpy random_state to use for random split; if None the default numpy random state is used
    :return: the product and a list if the products children
    """
    if proportions is not None:
        if num_parts is None:
            num_parts = len(proportions)
        else:
            if len(proportions) != num_parts:
                raise AssertionError("proportions should have num_parts elements.")
        proportions = np.array(proportions).astype(np.float64)
    else:
        proportions = np.ones(num_parts).astype(np.float64)

    if num_parts > len(node.scope):
        raise AssertionError(
            f"Cannot split scope of length {len(node.scope)} into {num_parts} parts."
        )

    proportions /= proportions.sum()
    if rand_state is not None:
        permutation = list(rand_state.permutation(list(node.scope)))
    else:
        permutation = list(np.random.permutation(list(node.scope)))

    child_indices = []
    for p in range(num_parts):
        p_len = int(np.round(len(permutation) * proportions[0]))
        p_len = min(max(p_len, 1), p + 1 + len(permutation) - num_parts)
        child_indices.append(permutation[0:p_len])
        permutation = permutation[p_len:]
        proportions = proportions[1:]
        proportions /= proportions.sum()

    return partition_on_node(graph, node, child_indices)


def random_binary_trees(num_var: int, depth: int, num_repetitions: int) -> DiGraph[DistributionVector]:
    """
    Generate a PC graph via several random binary trees -- RAT-SPNs.

    See
        Random sum-product networks: A simple but effective approach to probabilistic deep learning
        Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao, Martin Trapp, Kristian Kersting,
        Zoubin Ghahramani
        UAI 2019

    :param num_var: number of random variables (int)
    :param depth: splitting depth (int)
    :param num_repetitions: number of repetitions (int)
    :return: generated graph (DiGraph)
    """
    graph = nx.DiGraph()
    root = DistributionVector(range(num_var))
    graph.add_node(root)

    for repetition in range(num_repetitions):
        cur_nodes = [root]
        for d in range(depth):
            child_nodes = []
            for node in cur_nodes:
                _, cur_child_nodes = randomly_partition_on_node(graph, node, 2)
                child_nodes += cur_child_nodes
            cur_nodes = child_nodes
        for node in cur_nodes:
            node.einet_address.replica_idx = repetition

    return graph


def topological_layers(graph) -> list[list[DistributionVector]]:
    """
    Arranging the PC graph in topological layers -- see Algorithm 1 in the paper.

    :param graph: the PC graph (DiGraph)
    :return: list of layers, alternating between DistributionVector and Product layers (list of lists of nodes).
    """
    visited_nodes = set()
    layers = []

    sums = sorted(get_sums(graph))
    products = sorted(get_products(graph))
    leaves = sorted(get_leaves(graph))

    num_internal_nodes = len(sums) + len(products)

    while len(visited_nodes) != num_internal_nodes:
        sum_layer = [
            s
            for s in sums
            if s not in visited_nodes
            and all(p in visited_nodes for p in graph.predecessors(s))
        ]
        sum_layer = sorted(sum_layer)
        layers.insert(0, sum_layer)
        visited_nodes.update(sum_layer)

        product_layer = [
            p
            for p in products
            if p not in visited_nodes
            and all(s in visited_nodes for s in graph.predecessors(p))
        ]
        product_layer = sorted(product_layer)
        layers.insert(0, product_layer)
        visited_nodes.update(product_layer)

    layers.insert(0, leaves)
    return layers
