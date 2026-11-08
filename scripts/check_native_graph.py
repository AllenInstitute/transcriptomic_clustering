"""Exercise the native igraph/Leiden boundary before running the test suite."""
from importlib.metadata import version

import igraph
import leidenalg


def main():
    graph = igraph.Graph(
        n=6, edges=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    )
    partition = leidenalg.find_partition(
        graph, leidenalg.ModularityVertexPartition, seed=123
    )
    assert {frozenset(group) for group in partition} == {
        frozenset([0, 1, 2]), frozenset([3, 4, 5])
    }
    print(f"Native graph smoke passed: igraph {igraph.__version__}, "
          f"leidenalg {version('leidenalg')}")


if __name__ == "__main__":
    main()
