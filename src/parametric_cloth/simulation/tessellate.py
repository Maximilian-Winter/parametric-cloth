"""Turn a 2D pattern polygon into a triangulated cloth mesh.

Ear-clipping triangulates the (possibly non-convex) polygon; midpoint
subdivision then adds interior resolution so the cloth can drape and wrinkle.

Invariant: the original polygon corners are kept as mesh vertices ``0 ..
n_corners-1`` (subdivision only *appends* new midpoint vertices). This lets the
seam code (``seams.py``) map a ``SeamEdge`` -- expressed in original corner
indices -- onto the tessellated boundary without extra bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pattern import PatternPiece


@dataclass
class PieceMesh:
    """A tessellated pattern piece in flat 2D pattern space (centimeters)."""

    name: str
    vertices: np.ndarray              # (V, 2) cm; vertices[:n_corners] are corners
    faces: np.ndarray                 # (F, 3) int
    n_corners: int

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _point_in_triangle(p, a, b, c, *, eps: float = 1e-12) -> bool:
    """True if p is strictly inside triangle abc (any winding)."""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def ear_clip(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon, returning triangles as original-index triples."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Work counter-clockwise so the convex test has a consistent sign.
    idx = list(range(n))
    if _signed_area(points) < 0:
        idx.reverse()

    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        m = len(idx)
        ear_found = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = points[i0], points[i1], points[i2]
            # Convex corner (CCW)?
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            # No other vertex inside the candidate ear?
            if any(
                _point_in_triangle(points[p], a, b, c)
                for p in idx if p not in (i0, i1, i2)
            ):
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            ear_found = True
            break
        if not ear_found:   # degenerate polygon; bail with what we have
            break

    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def midpoint_subdivide(
    vertices: np.ndarray, faces: np.ndarray, levels: int
) -> tuple[np.ndarray, np.ndarray]:
    """1-to-4 split each triangle ``levels`` times; original vertices kept first."""
    verts = [np.asarray(v, dtype=float) for v in vertices]
    faces = [tuple(int(i) for i in f) for f in faces]

    for _ in range(max(0, levels)):
        cur = verts
        edge_mid: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in edge_mid:
                edge_mid[key] = len(verts)
                verts.append((cur[a] + cur[b]) / 2.0)
            return edge_mid[key]

        new_faces: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = new_faces

    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


def tessellate_piece(piece: PatternPiece, *, levels: int = 2) -> PieceMesh:
    """Tessellate one pattern piece into a ``PieceMesh``."""
    corners = np.array([[v.x, v.y] for v in piece.vertices], dtype=float)
    base_faces = ear_clip(corners)
    if not base_faces:
        raise ValueError(f"could not triangulate piece '{piece.name}'")
    vertices, faces = midpoint_subdivide(corners, base_faces, levels)
    return PieceMesh(
        name=piece.name, vertices=vertices, faces=faces, n_corners=len(corners)
    )
