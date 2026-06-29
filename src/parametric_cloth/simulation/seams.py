"""Seam vertex correspondence for pre-merge stitching (Strategy A).

A ``Seam`` connects two ``SeamEdge`` ranges, each given in *original corner
indices* of a piece. To weld the pieces we must (1) recover the tessellated
boundary loop of each piece, (2) map the corner range onto the chain of boundary
vertices (corners + inserted midpoints) it spans, and (3) pair each vertex on
one chain with the nearest vertex on the other.

All pure -- testable without Blender.
"""

from __future__ import annotations

import numpy as np


def boundary_loop(faces: np.ndarray) -> list[int]:
    """Ordered vertex indices around the single boundary of a triangulation.

    A directed edge is on the boundary when its reverse is absent (it belongs to
    exactly one triangle). Chaining those directed edges yields the loop.
    """
    directed: set[tuple[int, int]] = set()
    for a, b, c in (tuple(int(i) for i in f) for f in faces):
        directed.update({(a, b), (b, c), (c, a)})

    nxt = {a: b for (a, b) in directed if (b, a) not in directed}
    if not nxt:
        return []

    start = next(iter(nxt))
    loop = [start]
    cur = nxt[start]
    while cur != start and cur in nxt and len(loop) <= len(nxt):
        loop.append(cur)
        cur = nxt[cur]
    return loop


def seam_vertex_chain(
    loop: list[int], n_corners: int, start_corner: int, end_corner: int
) -> list[int]:
    """Boundary vertices spanning the polygon edge from ``start`` to ``end``.

    Walks the loop in whichever direction isolates a single polygon edge -- i.e.
    the chain whose interior contains no *other* original corner.
    """
    if start_corner not in loop or end_corner not in loop:
        raise ValueError("seam corner not present on boundary loop")

    forward = _walk(loop, start_corner, end_corner, step=+1)
    backward = _walk(loop, start_corner, end_corner, step=-1)

    def interior_has_corner(chain: list[int]) -> bool:
        return any(v < n_corners for v in chain[1:-1])

    f_clean = not interior_has_corner(forward)
    b_clean = not interior_has_corner(backward)
    if f_clean and not b_clean:
        return forward
    if b_clean and not f_clean:
        return backward
    # Both (or neither) clean: take the shorter span.
    return forward if len(forward) <= len(backward) else backward


def _walk(loop: list[int], start: int, end: int, *, step: int) -> list[int]:
    n = len(loop)
    i = loop.index(start)
    chain = [loop[i]]
    for _ in range(n):
        i = (i + step) % n
        chain.append(loop[i])
        if loop[i] == end:
            return chain
    return chain


def closest_point_pairs(
    positions_a: np.ndarray, positions_b: np.ndarray
) -> list[tuple[int, int]]:
    """For each row of A, the index of the nearest row of B.

    Returns ``(local_a_index, local_b_index)`` pairs (indices into the inputs).
    """
    a = np.asarray(positions_a, dtype=float)
    b = np.asarray(positions_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return []
    # Pairwise distances (chains are short, so the full matrix is cheap).
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    return [(int(i), int(j)) for i, j in enumerate(nearest)]
