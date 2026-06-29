"""UV generation from 2D pattern coordinates.

The pattern-based pipeline's key advantage: the original 2D pattern *is* the UV
map. Each mesh vertex already carries its flat pattern coordinate
(``AssembledGarment.pattern_uv``), so no unwrapping is needed -- each panel
becomes a UV island, packed into a single atlas. Textures painted on the flat
pattern then wrap correctly on the 3D garment, exactly like real fabric
printing.

Pure numpy -- fully testable without Blender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..simulation.assembly import AssembledGarment


@dataclass
class AtlasLayout:
    """UV coordinates (V, 2) in [0, 1] plus the grid the panels were packed in."""

    uv: np.ndarray
    cols: int
    rows: int
    margin: float

    def cell_of_panel(self, panel: int) -> tuple[float, float, float, float]:
        """(u_min, v_min, u_max, v_max) of the atlas cell for a panel index."""
        c, r = panel % self.cols, panel // self.cols
        cw, ch = 1.0 / self.cols, 1.0 / self.rows
        return (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)


def pack_uv_atlas(assembled: AssembledGarment, *, margin: float = 0.02) -> AtlasLayout:
    """Pack each panel's flat pattern coordinates into a single UV atlas.

    Panels are laid out in a near-square grid. Each panel is scaled uniformly
    (aspect ratio preserved, so textures are not distorted) to fit its cell with
    ``margin`` padding, and centered within it.
    """
    pattern = np.asarray(assembled.pattern_uv, dtype=float)
    panel_ids = np.asarray(assembled.vertex_panel)
    n_panels = assembled.n_panels

    uv = np.zeros((pattern.shape[0], 2), dtype=float)
    if n_panels == 0:
        return AtlasLayout(uv=uv, cols=0, rows=0, margin=margin)

    cols = math.ceil(math.sqrt(n_panels))
    rows = math.ceil(n_panels / cols)
    cell = np.array([1.0 / cols, 1.0 / rows])
    margin_abs = margin * cell
    avail = cell - 2.0 * margin_abs

    for p in range(n_panels):
        idx = np.where(panel_ids == p)[0]
        if idx.size == 0:
            continue
        pts = pattern[idx]
        lo = pts.min(axis=0)
        size = pts.max(axis=0) - lo
        size[size == 0] = 1.0                       # guard degenerate axis

        scale = float(np.min(avail / size))         # uniform -> aspect preserved
        used = size * scale
        origin = np.array([(p % cols) / cols, (p // cols) / rows])
        centering = (avail - used) / 2.0
        uv[idx] = origin + margin_abs + centering + (pts - lo) * scale

    return AtlasLayout(uv=uv, cols=cols, rows=rows, margin=margin)
