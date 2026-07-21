"""Weld paired vertices in an assembled mesh (pure NumPy, union-find).

This is the same operation Blender's ``bmesh.ops.remove_doubles`` performs
after :func:`~parametric_cloth.simulation.blender_sim.weld_seams` snaps seam
pairs together -- implemented here in plain NumPy so a fully assembled, seamed
garment mesh can be built and drape-tested without Blender.
"""

from __future__ import annotations

import numpy as np


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def weld_vertices(
    vertices: np.ndarray, faces: np.ndarray, pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge each ``(i, j)`` pair into one vertex (at their midpoint).

    Pairs chain transitively via union-find, so a vertex welded to two
    different partners (e.g. three panels meeting at a corner) still merges
    correctly into a single point. Degenerate faces created by welding (a
    triangle whose corners collapsed together) are dropped.

    Returns ``(new_vertices, new_faces, old_to_new)`` where ``old_to_new`` maps
    every original vertex index to its compacted index in ``new_vertices``.
    """
    n = len(vertices)
    uf = _UnionFind(n)
    for i, j in pairs:
        uf.union(int(i), int(j))

    roots = np.array([uf.find(i) for i in range(n)])
    _, old_to_new = np.unique(roots, return_inverse=True)

    n_new = int(old_to_new.max()) + 1 if n else 0
    new_vertices = np.zeros((n_new, vertices.shape[1]))
    counts = np.zeros(n_new)
    np.add.at(new_vertices, old_to_new, vertices)
    np.add.at(counts, old_to_new, 1.0)
    new_vertices /= counts[:, None]

    new_faces = old_to_new[faces]
    degenerate = (
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 1] == new_faces[:, 2])
        | (new_faces[:, 2] == new_faces[:, 0])
    )
    new_faces = new_faces[~degenerate]

    return new_vertices, new_faces, old_to_new
